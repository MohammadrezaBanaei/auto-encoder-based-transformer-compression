from typing import List

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, input: torch.tensor):
        raise NotImplementedError

    def get_latent_dim(self) -> int:
        return self.latent_dim


class LinearEncoder(Encoder):
    def __init__(self, input_dim: int, latent_dim: int):
        super(LinearEncoder, self).__init__(input_dim=input_dim, latent_dim=latent_dim)
        self.enc = nn.Linear(in_features=input_dim, out_features=latent_dim)

    def forward(self, input: torch.tensor):
        return self.enc(input)


class NonLinearEncoder(Encoder):
    def __init__(self, input_dim: int, latent_dim: int, enc_layer_sizes: List):
        super(NonLinearEncoder, self).__init__(input_dim=input_dim, latent_dim=latent_dim)
        self.activation = nn.LeakyReLU()
        self.enc = nn.ModuleList()
        enc_layer_sizes = [input_dim] + enc_layer_sizes
        for i in range(0, len(enc_layer_sizes)-1):
            self.enc.append(nn.Linear(in_features=enc_layer_sizes[i], out_features=enc_layer_sizes[i+1]))
            self.enc.append(self.activation)
        self.enc.append(nn.Linear(in_features=enc_layer_sizes[-1], out_features=latent_dim))
        self.enc.append(self.activation)

    def forward(self, input: torch.tensor):
        for m in self.enc:
            input = m(input)
        return input
