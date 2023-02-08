import os
from typing import List

import torch
from torch.utils.data import Dataset


class MatrixDataset(Dataset):
    def __init__(self, matrix: torch.tensor, row_ids: List, coefficients: torch.Tensor = None):
        self.matrix = matrix
        self.row_ids = row_ids
        self.coefficients = coefficients

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, idx):
        if self.coefficients is None:
            return self.matrix[idx], self.row_ids[idx]
        return self.matrix[idx], self.row_ids[idx], self.coefficients[idx]


class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states_tensor: torch.tensor, module_name: str, modules_training_setting_name: str):
        # hidden_states_tensor has shape of (#transformer_layers + 1, num_samples, hidden_dim) which
        # includes the inputs to all the transformer layers plus the model output

        if modules_training_setting_name == 'separated':
            layer_num = int(module_name.split("_")[1])
            self.matrix = hidden_states_tensor[layer_num]
        elif modules_training_setting_name == 'concatenated':
            _, num_samples, hidden_dim = hidden_states_tensor.shape
            self.matrix = hidden_states_tensor[:-1].transpose(0, 1)
        else:
            raise ValueError(f'No such modules_training_setting as {modules_training_setting_name}.')

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        return self.matrix[idx]
