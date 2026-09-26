import os
import sys

# Add project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.core.models import MRAUNet
from src.training.dataset import BraTSDataset
from src.training.loss import CombinedBraTSLoss
from src.training.metrics import compute_dice_score, compute_hd95


def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    use_cuda = (device == "cuda" or "cuda" in str(device))

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 25 == 0 or (i + 1) == num_batches:
            pct = int(((i + 1) / num_batches) * 100)
            print(f"  Batch [{i+1}/{num_batches}] ({pct}%) - Loss: {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_wt_total = 0.0
    hd95_wt_total = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = targets.cpu().numpy()

            for b in range(preds.shape[0]):
                pred_wt = (preds[b] > 0).astype(np.uint8)
                gt_wt = (gts[b] > 0).astype(np.uint8)

                dice_wt_total += compute_dice_score(pred_wt, gt_wt)
                hd95_wt_total += compute_hd95(pred_wt, gt_wt, voxel_spacing=(1.0, 1.0, 1.0))

    total_samples = max(len(dataloader.dataset), 1)
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = dice_wt_total / total_samples
    avg_hd95 = hd95_wt_total / total_samples
    return avg_loss, avg_dice, avg_hd95


def main():
    parser = argparse.ArgumentParser(description="Train MRAU-Net on Real Brain MRI Dataset")
    parser.add_argument("--data_dir", type=str, default="./datasets/BraTS2024", help="Path to directory containing patient NIfTI folders")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.20, help="Validation set split fraction")
    parser.add_argument("--max_patients", type=int, default=0, help="Max number of patients to use (0 = all)")
    parser.add_argument("--slice_step", type=int, default=3, help="Step between axial slices (e.g. 3 or 4)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"🚀 Initializing MRAU-Net Training Pipeline on {args.device}...")
    print(f"📂 Dataset Directory: {args.data_dir}")

    # Check if dataset directory exists
    if not os.path.exists(args.data_dir):
        print(f"\n⚠️ Directory '{args.data_dir}' not found.")
        print("Please structure your dataset as follows:")
        print("datasets/BraTS2024/")
        print("  ├── Patient_001/")
        print("  │   ├── *t1*.nii.gz")
        print("  │   ├── *t1ce*.nii.gz")
        print("  │   ├── *t2*.nii.gz")
        print("  │   ├── *flair*.nii.gz")
        print("  │   └── *seg*.nii.gz")
        print("  └── Patient_002/ ...")
        return

    # 1. Dataset & DataLoader
    full_dataset = BraTSDataset(
        data_dir=args.data_dir,
        mode="2d_slices",
        transform=True,
        slice_step=args.slice_step,
        max_patients=args.max_patients if args.max_patients > 0 else None
    )
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True if "cuda" in args.device else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True if "cuda" in args.device else False
    )

    print(f"📊 Total Samples: {len(full_dataset)} | Training: {train_size} | Validation: {val_size}")

    # 2. Model, Optimizer, Scaler, Loss
    model = MRAUNet(in_channels=4, num_classes=4).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=("cuda" in args.device))
    criterion = CombinedBraTSLoss().to(args.device)

    best_dice = 0.0

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        print(f"\n Epoch [{epoch}/{args.epochs}] (lr: {optimizer.param_groups[0]['lr']:.6f})")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, args.device)
        val_loss, val_dice, val_hd95 = validate_epoch(model, val_loader, criterion, args.device)
        scheduler.step()

        elapsed = time.time() - start_t
        print(f" Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WT Dice: {val_dice:.4f} | Val WT HD95: {val_hd95:.2f}mm | Time: {elapsed:.1f}s")

        # Save Best Model Checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(args.save_dir, "best_mraunet.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_hd95": val_hd95
            }, save_path)
            print(f" 💾 Saved new best model checkpoint to {save_path} (WT Dice: {val_dice:.4f} | WT HD95: {val_hd95:.2f}mm)")

    print(f"\n🎉 Training complete! Best Validation WT Dice Score: {best_dice:.4f}")


if __name__ == "__main__":
    main()
