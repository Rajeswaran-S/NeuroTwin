"""
Loss Functions for Multi-Class Brain Tumor Segmentation
Implements Combined Dice Loss + Cross-Entropy Loss (from Paper 1 & Paper 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss for BraTS sub-compartments:
    - Channel 0: Background
    - Channel 1: Necrotic / Non-enhancing Core (NET)
    - Channel 2: Peritumoral Edema (ED)
    - Channel 3: Enhancing Tumor (ET)
    """
    def __init__(self, smooth: float = 1e-5, weights: list = [0.1, 1.0, 1.0, 1.5]):
        super().__init__()
        self.smooth = smooth
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); targets: (B, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        total_loss = 0.0
        for c in range(num_classes):
            p_c = probs[:, c, :, :].contiguous().view(-1)
            t_c = targets_one_hot[:, c, :, :].contiguous().view(-1)

            intersection = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum()
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            w = self.weights[c] if c < len(self.weights) else 1.0
            total_loss += w * (1.0 - dice_score)

        return total_loss / sum(self.weights)


class CombinedBraTSLoss(nn.Module):
    """
    Composite Loss = Cross-Entropy Loss + Dice Loss
    """
    def __init__(self, dice_weight: float = 0.6, ce_weight: float = 0.4):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = MultiClassDiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0, 1.5]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        if self.ce_loss.weight.device != device:
            self.ce_loss.weight = self.ce_loss.weight.to(device)
            
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice
