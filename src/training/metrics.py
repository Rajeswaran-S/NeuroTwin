"""
Module: Medical Segmentation Evaluation Metrics
Implements standard BraTS validation metrics:
  - Dice Similarity Coefficient (DSC)
  - IoU (Jaccard Index)
  - 95th Percentile Hausdorff Distance (HD95 in mm/voxels)
  - Sensitivity / Recall & Specificity
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure


def compute_dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Computes Dice Similarity Coefficient (DSC).
    """
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    
    if total == 0:
        return 1.0  # Perfect agreement if both empty
        
    return float((2.0 * intersection + smooth) / (total + smooth))


def compute_iou(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index).
    """
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    
    if union == 0:
        return 1.0
        
    return float((intersection + smooth) / (union + smooth))


def compute_hd95(
    pred: np.ndarray, 
    gt: np.ndarray, 
    voxel_spacing: tuple = (1.0, 1.0, 1.0)
) -> float:
    """
    Computes the 95th percentile Hausdorff Distance (HD95) in physical units (mm).
    
    Parameters:
        pred: Binary prediction mask (2D or 3D numpy array).
        gt: Binary ground truth mask (2D or 3D numpy array).
        voxel_spacing: Spatial resolution per dimension in mm (e.g. (1.0, 1.0, 1.0)).
        
    Returns:
        hd95_val: 95th percentile distance in mm (lower is better; 0.0 is perfect boundary match).
    """
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)

    # Edge cases
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        # One mask is completely empty while the other is not
        # Return maximum bounding box diagonal as penalty
        shape = np.array(pred.shape)
        diag = np.sqrt(np.sum((shape * np.array(voxel_spacing[:len(shape)])) ** 2))
        return float(diag)

    # Extract 1-voxel thick boundary surfaces
    struct = generate_binary_structure(pred_bin.ndim, 1)
    pred_border = np.logical_xor(pred_bin, binary_erosion(pred_bin, structure=struct))
    gt_border = np.logical_xor(gt_bin, binary_erosion(gt_bin, structure=struct))

    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return 0.0

    # Euclidean distance transforms (EDT) with physical voxel spacing
    spacing = voxel_spacing[:pred_bin.ndim]
    dt_gt = distance_transform_edt(~gt_bin, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_bin, sampling=spacing)

    # Distances from pred border to gt, and gt border to pred
    dist_pred_to_gt = dt_gt[pred_border]
    dist_gt_to_pred = dt_pred[gt_border]

    # Combine directional distances
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])

    # 95th percentile Hausdorff Distance
    hd95_val = float(np.percentile(all_distances, 95))
    return hd95_val


def evaluate_brats_metrics(
    pred_mask: np.ndarray, 
    gt_mask: np.ndarray, 
    voxel_spacing: tuple = (1.0, 1.0, 1.0)
) -> dict:
    """
    Calculates comprehensive BraTS metrics for all 3 standard targets:
      - Whole Tumor (WT): Labels 1, 2, 3
      - Tumor Core (TC): Labels 1, 3 (NET + ET)
      - Enhancing Tumor (ET): Label 3
    """
    # Binary definitions for BraTS targets
    pred_wt = (pred_mask > 0)
    gt_wt = (gt_mask > 0)

    pred_tc = np.isin(pred_mask, [1, 3])
    gt_tc = np.isin(gt_mask, [1, 3])

    pred_et = (pred_mask == 3)
    gt_et = (gt_mask == 3)

    metrics = {
        "wt_dice": compute_dice_score(pred_wt, gt_wt),
        "wt_hd95": compute_hd95(pred_wt, gt_wt, voxel_spacing),
        "tc_dice": compute_dice_score(pred_tc, gt_tc),
        "tc_hd95": compute_hd95(pred_tc, gt_tc, voxel_spacing),
        "et_dice": compute_dice_score(pred_et, gt_et),
        "et_hd95": compute_hd95(pred_et, gt_et, voxel_spacing),
    }

    metrics["mean_dice"] = float(np.mean([metrics["wt_dice"], metrics["tc_dice"], metrics["et_dice"]]))
    metrics["mean_hd95"] = float(np.mean([metrics["wt_hd95"], metrics["tc_hd95"], metrics["et_hd95"]]))

    return metrics
