import torch
from torch import nn
from typing import Optional


def l2_norm_loss(output: torch.tensor, target: torch.tensor, epoch: int, rec_coef: Optional[torch.Tensor] = None):
    mse_loss = nn.MSELoss(reduction='none')
    mse_loss_value = mse_loss(output, target)
    if rec_coef is not None:
        if len(rec_coef.shape) == 1:
            mse_loss_value = mse_loss_value.mean(-1) * rec_coef
        else:
            mse_loss_value = mse_loss_value * rec_coef
    rmse_loss_value = torch.sqrt(mse_loss_value.mean())
    return rmse_loss_value
