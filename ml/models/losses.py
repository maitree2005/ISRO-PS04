import torch
import torch.nn as nn
import torch.nn.functional as F
from ml.models.topology import ContinuityLoss


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W) with values 0..(C-1)
        if logits.dim() != 4:
            raise ValueError('Logits must be 4D')
        probs = torch.softmax(logits, dim=1)
        if probs.size(1) == 1:
            probs_fg = torch.sigmoid(logits).squeeze(1)
            targets_f = targets.float()
        else:
            probs_fg = probs[:, 1, :, :]
            targets_f = (targets == 1).float()

        probs_flat = probs_fg.contiguous().view(probs_fg.size(0), -1)
        targets_flat = targets_f.contiguous().view(targets_f.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        return dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.topology_weight = 0.0
        self.topology_loss = None

    def enable_topology(self, weight: float = 1.0, kernel_size: int = 5):
        self.topology_weight = float(weight)
        if self.topology_weight > 0.0:
            self.topology_loss = ContinuityLoss(kernel_size=kernel_size)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        if self.topology_loss is not None and self.topology_weight > 0.0:
            topo = self.topology_loss(logits, targets)
            loss = loss + self.topology_weight * topo
        return loss
