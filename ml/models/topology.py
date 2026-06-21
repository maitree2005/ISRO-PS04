import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuityLoss(nn.Module):
    """A differentiable surrogate that encourages local continuity in predicted probabilities.

    For each ground-truth foreground pixel, we require at least one high predicted
    probability in its local neighborhood. Implemented via a max-pooling of the
    predicted foreground probability and penalizing (1 - local_max) inside the mask.
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        if logits.dim() != 4:
            raise ValueError('Logits must be 4D')

        if logits.size(1) == 1:
            probs = torch.sigmoid(logits).squeeze(1)
        else:
            probs = torch.softmax(logits, dim=1)[:, 1, :, :]

        targets_f = (targets == 1).float()

        # local max in a neighborhood
        local_max = self.pool(probs)
        loss_map = (1.0 - local_max) * targets_f
        return loss_map.mean()
