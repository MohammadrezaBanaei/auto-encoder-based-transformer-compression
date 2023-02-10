from typing import List

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int, org_norm: torch.tensor, finetune_org_norm: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.org_norm = org_norm
        if finetune_org_norm:
            self.detach_org_norm_and_enable_training()

    def forward(self, input: torch.tensor, ids: torch.tensor):
        raise NotImplementedError

    def get_decoder_size(self) -> int:
        raise NotImplementedError

    def detach_org_norm_and_enable_training(self):
        if self.org_norm is not None:
            self.org_norm = self.org_norm.detach()
            self.org_norm.requires_grad = True


class LinearDecoder(Decoder):
    def __init__(self, out_dim: int, latent_dim: int, org_norm: torch.tensor, finetune_org_norm: bool = True):
        super(LinearDecoder, self).__init__(out_dim=out_dim, latent_dim=latent_dim, org_norm=org_norm,
                                            finetune_org_norm=finetune_org_norm)
        self.dec = nn.Linear(in_features=latent_dim, out_features=out_dim)

    def forward(self, input: torch.tensor, ids: torch.tensor):
        input = self.dec(input)
        if self.org_norm is not None:
            input_norm = torch.linalg.norm(input, dim=-1)
            input = torch.div(input, (input_norm / self.org_norm[ids]).unsqueeze(-1))
        return input

    def get_decoder_size(self) -> int:
        return self.dec.weight.numel() + self.dec.bias.numel()


class NonLinearDecoder(Decoder):
    def __init__(self, out_dim: int, latent_dim: int, dec_layer_sizes: List,
                 org_norm: torch.tensor, finetune_org_norm: bool = True):
        super(NonLinearDecoder, self).__init__(out_dim=out_dim, latent_dim=latent_dim, org_norm=org_norm,
                                               finetune_org_norm=finetune_org_norm)
        self.activation = nn.LeakyReLU()
        self.final_activation = nn.Tanh()
        self.dec = nn.ModuleList()

        dec_layer_sizes = dec_layer_sizes + [out_dim]
        self.dec.append(nn.Linear(in_features=latent_dim, out_features=dec_layer_sizes[0]))
        for i in range(len(dec_layer_sizes)-1):
            self.dec.append(self.activation)
            self.dec.append(nn.Linear(in_features=dec_layer_sizes[i], out_features=dec_layer_sizes[i+1]))
        self.dec.append(self.final_activation)

    def forward(self, input: torch.tensor, ids: torch.tensor):
        for m in self.dec:
            input = m(input)
        if self.org_norm is not None:
            input_norm = torch.linalg.norm(input, dim=-1)
            input = torch.div(input, (input_norm/self.org_norm[ids]).unsqueeze(-1))
        return input

    def get_decoder_size(self) -> int:
        params_num = 0
        for l in self.dec:
            if type(l) == nn.Linear:
                params_num += (l.weight.numel() + l.bias.numel())
        return params_num

