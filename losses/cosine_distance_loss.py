import torch
from torch import nn
from typing import Optional


def cosine_distance_loss(output: torch.tensor, target: torch.tensor, epoch: int, dim: int = -1,
                         rec_coef: Optional[torch.Tensor] = None):
    cos = nn.CosineSimilarity(dim=dim, eps=1e-6)
    cos_loss = 1.0 - cos(output, target)
    if rec_coef is not None:
        cos_loss = cos_loss * rec_coef
    cos_loss = cos_loss.mean()
    return cos_loss
