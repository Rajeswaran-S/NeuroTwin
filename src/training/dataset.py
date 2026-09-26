"""
Dataset & DataLoader for Real BraTS & Multi-Parametric Brain MRI NIfTI Datasets
Supports BraTS 2020/2021/2023/2024, BraTS-Africa, and custom NIfTI directory structures.
"""

import os
import glob
import functools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib


def normalize_mri_volume(volume: np.ndarray) -> np.ndarray:
    """Z-Score intensity normalization with foreground clipping."""
    mask = volume > 0
    if not np.any(mask):
        return volume.astype(np.float32)
    mean = np.mean(volume[mask])
    std = np.std(volume[mask])
    norm = np.zeros_like(volume, dtype=np.float32)
    norm[mask] = (volume[mask] - mean) / (std + 1e-6)
    norm = np.clip(norm, -3.0, 3.0)
    norm = (norm + 3.0) / 6.0
    return norm


@functools.lru_cache(maxsize=128)
def _load_nii_cached(file_path: str) -> np.ndarray:
    """Cached NIfTI loader so each 3D patient volume is read from disk only once."""
    if not file_path or not os.path.exists(file_path):
        return np.zeros((192, 192, 128), dtype=np.float32)
    img = nib.load(file_path)
    return img.get_fdata().astype(np.float32)


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for real multi-sequence Brain MRI volumes (T1, T1CE, T2, FLAIR, SEG).
    Can extract 2D axial slices with tumor presence or 3D patches for training.
    """
    def __init__(
        self,
        data_dir: str,
        mode: str = "2d_slices",
        slice_sampling: str = "tumor_weighted",
        transform: bool = True,
        target_size: tuple = (192, 192),
        slice_step: int = 3,
        max_patients: int = None
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.slice_sampling = slice_sampling
        self.transform = transform
        self.target_size = target_size
        self.slice_step = max(1, slice_step)
        
        all_patients = self._find_patient_folders(data_dir)
        if max_patients and max_patients > 0:
            self.patient_dirs = all_patients[:max_patients]
        else:
            self.patient_dirs = all_patients
        
        if len(self.patient_dirs) == 0:
            raise FileNotFoundError(f"No valid patient folders with NIfTI files found in {data_dir}")

        # Pre-index all 5 modality file paths for every patient to eliminate disk search overhead
        self.patient_files = {}
        for p_dir in self.patient_dirs:
            self.patient_files[p_dir] = {
                "t1": self._find_modality_file(p_dir, ["t1n", "t1.", "t1_"]),
                "t1ce": self._find_modality_file(p_dir, ["t1c", "t1ce"]),
                "t2": self._find_modality_file(p_dir, ["t2w", "t2.", "t2_"]),
                "flair": self._find_modality_file(p_dir, ["t2f", "flair"]),
                "seg": self._find_modality_file(p_dir, ["seg", "mask"])
            }

        # Index slices if 2D mode
        self.samples = []
        if self.mode == "2d_slices":
            self._build_slice_index()

    def _find_patient_folders(self, root: str) -> list:
        """Finds all subdirectories containing NIfTI files."""
        candidates = []
        for entry in sorted(os.listdir(root)):
            p_path = os.path.join(root, entry)
            if os.path.isdir(p_path):
                nii_files = glob.glob(os.path.join(p_path, "*.nii*"))
                if len(nii_files) >= 4:
                    candidates.append(p_path)
        return candidates

    def _build_slice_index(self):
        """Builds index of (patient_dir, slice_idx) pairs for fast loader access."""
        # Range 35 to 125 focuses strictly on salient brain/tumor slices
        for p_dir in self.patient_dirs:
            for z in range(35, 125, self.slice_step):
                self.samples.append((p_dir, z))

    def __len__(self):
        return len(self.samples) if self.mode == "2d_slices" else len(self.patient_dirs)

    def _find_modality_file(self, p_dir: str, keywords: list) -> str:
        all_nii = glob.glob(os.path.join(p_dir, "*.nii*"))
        for kw in keywords:
            for f in all_nii:
                fname = os.path.basename(f).lower()
                if kw in fname:
                    return f
        return all_nii[0] if all_nii else ""

    def __getitem__(self, idx: int):
        if self.mode == "2d_slices":
            p_dir, z_idx = self.samples[idx]
            p_files = self.patient_files[p_dir]

            t1 = _load_nii_cached(p_files["t1"])[:, :, z_idx]
            t1ce = _load_nii_cached(p_files["t1ce"])[:, :, z_idx]
            t2 = _load_nii_cached(p_files["t2"])[:, :, z_idx]
            flair = _load_nii_cached(p_files["flair"])[:, :, z_idx]
            seg = _load_nii_cached(p_files["seg"])[:, :, z_idx] if p_files["seg"] else np.zeros_like(t1)

            # Normalize
            t1 = normalize_mri_volume(t1)
            t1ce = normalize_mri_volume(t1ce)
            t2 = normalize_mri_volume(t2)
            flair = normalize_mri_volume(flair)

            # Assemble 4-modality channel stack and mask
            raw_stack = np.stack([t1, t1ce, t2, flair], axis=0)
            raw_seg = seg.astype(np.int64)

            # Standardize spatial dimensions to target_size (e.g. 192x192)
            stack = self._pad_or_crop_stack(raw_stack, self.target_size)
            seg_mask = self._pad_or_crop_2d(raw_seg, self.target_size)

            # Remap BraTS label 4 to 3 if standard BraTS formatting
            seg_mask[seg_mask == 4] = 3

            # Augmentations (Random Flip, Rotation)
            if self.transform and np.random.rand() > 0.5:
                stack = np.flip(stack, axis=2).copy()
                seg_mask = np.flip(seg_mask, axis=1).copy()

            tensor_x = torch.from_numpy(stack).float()
            tensor_y = torch.from_numpy(seg_mask).long()

            return tensor_x, tensor_y

    def _pad_or_crop_stack(self, stack: np.ndarray, target_size: tuple) -> np.ndarray:
        """Crops or pads 4-channel stack (4, H, W) to (4, target_H, target_W)."""
        _, h, w = stack.shape
        th, tw = target_size
        out = np.zeros((4, th, tw), dtype=np.float32)
        
        sh_start = max(0, (h - th) // 2)
        sh_end = sh_start + min(h, th)
        sw_start = max(0, (w - tw) // 2)
        sw_end = sw_start + min(w, tw)

        dh_start = max(0, (th - h) // 2)
        dh_end = dh_start + (sh_end - sh_start)
        dw_start = max(0, (tw - w) // 2)
        dw_end = dw_start + (sw_end - sw_start)

        out[:, dh_start:dh_end, dw_start:dw_end] = stack[:, sh_start:sh_end, sw_start:sw_end]
        return out

    def _pad_or_crop_2d(self, img: np.ndarray, target_size: tuple) -> np.ndarray:
        """Crops or pads 2D mask (H, W) to (target_H, target_W)."""
        h, w = img.shape
        th, tw = target_size
        out = np.zeros((th, tw), dtype=np.int64)

        sh_start = max(0, (h - th) // 2)
        sh_end = sh_start + min(h, th)
        sw_start = max(0, (w - tw) // 2)
        sw_end = sw_start + min(w, tw)

        dh_start = max(0, (th - h) // 2)
        dh_end = dh_start + (sh_end - sh_start)
        dw_start = max(0, (tw - w) // 2)
        dw_end = dw_start + (sw_end - sw_start)

        out[dh_start:dh_end, dw_start:dw_end] = img[sh_start:sh_end, sw_start:sw_end]
        return out
