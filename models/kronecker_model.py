import math
from typing import Tuple

import torch
import torch.nn as nn


class KroneckerProductModel(nn.Module):
    def __init__(self, r: int, original_matrix_shape: int):
        super().__init__()
        self.n = original_matrix_shape[0]
        self.m = original_matrix_shape[1]
        self.sqrt_n = math.ceil(math.sqrt(original_matrix_shape[0]))
        self.sqrt_m = math.ceil(math.sqrt(original_matrix_shape[1]))

        sqrt_nm = self.sqrt_n * self.sqrt_m
        self.a = nn.Parameter(torch.empty((sqrt_nm, r)), requires_grad=True)
        self.b = nn.Parameter(torch.empty((r, sqrt_nm)), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Code from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.b, a=math.sqrt(5))

    def forward(self, x, ids) -> Tuple[torch.Tensor, None]:
        # We take mean per ids
        res = torch.matmul(self.a, self.b).reshape(self.sqrt_n, self.sqrt_m, self.sqrt_n, self.sqrt_m) \
                   .permute(0, 2, 1, 3).reshape((self.sqrt_n) ** 2, (self.sqrt_m) ** 2)[:self.n, :self.m]

        return res[ids, :], None

    def get_substitution_module_size(self) -> int:
        return self.a.numel() + self.b.numel()
