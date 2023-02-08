from typing import Dict, Optional

import torch
from torch import nn


def l1_norm_loss(output: torch.tensor, target: torch.tensor, epoch: int, config: Dict, changing_epochs_num: int,
                 epsilon: float = 1e-10,  rec_coef: Optional[torch.Tensor] = None):
    start_alpha = config['start_alpha']
    end_alpha = config['end_alpha']
    if epoch >= changing_epochs_num:
        alpha_value = end_alpha
    else:
        alpha_value = start_alpha - ((start_alpha-end_alpha) * (epoch / changing_epochs_num))

    l1_loss = nn.L1Loss(reduction='none')
    l1_loss_value = torch.pow(l1_loss(output, target) + epsilon, alpha_value)
    if rec_coef is not None:
        if len(rec_coef.shape) == 1:
            l1_loss_value = l1_loss_value.mean(-1) * rec_coef
        else:
            l1_loss_value = l1_loss_value * rec_coef
    l1_loss_value = l1_loss_value.mean()
    return l1_loss_value
