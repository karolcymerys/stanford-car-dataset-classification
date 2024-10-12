import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1 , gamma: float = 2) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy_loss = self.cross_entropy_loss(inputs, targets)
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * cross_entropy_loss
        return focal_loss.mean()
