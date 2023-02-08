from typing import List, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.utils_funcs import (get_ae_sub_compression_stats,
                               get_svd_compression_stats,
                               get_svd_prunning_mask,
                               get_lin_rec_latent_dim,
                               run_svd,
                               get_zeros_frac_for_svd)


class SubstitutionModule(nn.Module):
    def __init__(self,
                 original_params_num: int,
                 decoder_module: nn.Module = None):
        super().__init__()
        self.original_params_num = original_params_num
        self.decoder_module = decoder_module

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_compression_stats(self) -> Tuple[float, float]:
        raise NotImplementedError


class SubstitutionWeightModule(SubstitutionModule):
    def __init__(self,
                 decoder_module: nn.Module,
                 latent_weights: torch.Tensor,
                 original_params_num: int,
                 module_bias: torch.Tensor):
        super().__init__(original_params_num=original_params_num,
                         decoder_module=decoder_module)
        self.latent_weights = nn.Parameter(latent_weights)
        self.module_bias = nn.Parameter(module_bias)
        self.latent_ids = torch.tensor(range(len(latent_weights)), device=latent_weights.device)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        temp_projection_matrix = self.decoder_module(self.latent_weights, self.latent_ids).T
        output = input_ @ temp_projection_matrix + self.module_bias
        return output

    def get_compression_stats(self) -> Tuple[float, float]:
        cr, cr_with_zeros, _, _, _ = get_ae_sub_compression_stats(original_params_num=self.original_params_num,
                                                                  embedding=self.latent_weights,
                                                                  decoder=self.decoder_module)
        return cr, cr_with_zeros


class WordEmbeddingModule(SubstitutionModule):

    def __init__(self,
                 decoder_module: nn.Module,
                 emb_weights: torch.Tensor,
                 pad_token_id: int,
                 original_params_num: int):
        super().__init__(original_params_num=original_params_num,
                         decoder_module=decoder_module)

        self.pad_token_id = pad_token_id
        self.embedding_module = nn.Embedding.from_pretrained(emb_weights,
                                                             padding_idx=self.pad_token_id,
                                                             freeze=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embedding_module(input_ids)

        inputs_embeds = WordEmbeddingModule.transform_emb_weights(inputs_embeds)

        out = self.decoder_module(inputs_embeds, input_ids)

        return out

    def get_compression_stats(self) -> Tuple[float, float]:
        emb_weights = WordEmbeddingModule.transform_emb_weights(self.embedding_module.weight)
        cr, cr_with_zeros, _, _, _ = get_ae_sub_compression_stats(original_params_num=self.original_params_num,
                                                                  embedding=emb_weights,
                                                                  decoder=self.decoder_module)
        return cr, cr_with_zeros

    @staticmethod
    def transform_emb_weights(emb_weights: torch.Tensor):
        return emb_weights


class MFWordEmbeddingModule(SubstitutionModule):

    def __init__(self,
                 reduced_matrix: torch.Tensor,
                 second_matrix: torch.Tensor,
                 pad_token_id: int,
                 original_params_num: int):
        super().__init__(original_params_num=original_params_num)

        self.pad_token_id = pad_token_id
        self.embedding_module = nn.Embedding.from_pretrained(reduced_matrix, padding_idx=self.pad_token_id)
        self.second_matrix = second_matrix

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embedding_module(input)
        svd_w = self.second_matrix

        result = torch.mm(inputs_embeds.view(-1, inputs_embeds.shape[2]), svd_w)
        result = result.view(-1, inputs_embeds.shape[1], self.second_matrix.shape[1])

        return result

    def get_compression_stats(self) -> Tuple[float, float]:
        return get_svd_compression_stats(orginal_params_num=self.original_params_num,
                                         reduced_matrix_numel=self.embedding_module.weight.numel(),
                                         second_matrix_numel=self.second_matrix.numel())

    @staticmethod
    def get_svd_emb_module(input_matrix: torch.Tensor,
                           compression_ratio: float,
                           n_iter: int,
                           random_state: int,
                           device: torch.device,
                           pad_token_id: int,
                           coefs: Optional[torch.Tensor] = None) -> SubstitutionModule:
        original_params_num = input_matrix.numel()
        latent_dim = get_lin_rec_latent_dim(compression_ratio, input_matrix)

        if coefs is not None:
            coefs = coefs.cpu().numpy()

        reduced_matrix, svd = run_svd(input_matrix=input_matrix.cpu().numpy(),
                                      latent_dim=latent_dim,
                                      n_iter=n_iter,
                                      random_state=random_state,
                                      rec_coef=coefs)

        reduced_matrix = torch.tensor(reduced_matrix, dtype=input_matrix.dtype, device=device)
        second_matrix = torch.tensor(svd.components_, dtype=input_matrix.dtype, device=device)

        svd_word_emb_module = MFWordEmbeddingModule(reduced_matrix=reduced_matrix,
                                                    second_matrix=second_matrix,
                                                    pad_token_id=pad_token_id,
                                                    original_params_num=original_params_num)
        return svd_word_emb_module

class MFWeightModule(SubstitutionModule):
    def __init__(self,
                 latent_weights: torch.Tensor,
                 module_bias: torch.Tensor,
                 second_matrix: torch.Tensor,
                 original_params_num: int):
        super().__init__(original_params_num=original_params_num)

        self.latent_weights = nn.Parameter(latent_weights)
        self.module_bias = nn.Parameter(module_bias)
        self.second_matrix = nn.Parameter(second_matrix)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        temp_projection_matrix = torch.mm(self.latent_weights, self.second_matrix).T
        output = input @ temp_projection_matrix + self.module_bias
        return output

    def get_compression_stats(self) -> Tuple[float, float]:
        return get_svd_compression_stats(orginal_params_num=self.original_params_num,
                                         reduced_matrix_numel=self.latent_weights.numel(),
                                         second_matrix_numel=self.second_matrix.numel())

    @staticmethod
    def get_svd_mf_module(input_matrix: torch.Tensor,
                          module_bias: torch.Tensor,
                          compression_ratio: float,
                          n_iter: int,
                          random_state: int,
                          device: torch.device,
                          coefs: Optional[torch.Tensor] = None) -> SubstitutionModule:
        original_params_num = input_matrix.numel()
        latent_dim = get_lin_rec_latent_dim(compression_ratio, input_matrix)

        if coefs is not None:
            coefs = coefs.cpu().numpy()

        reduced_matrix, svd = run_svd(input_matrix=input_matrix.cpu().numpy(),
                                      latent_dim=latent_dim,
                                      n_iter=n_iter,
                                      random_state=random_state,
                                      rec_coef=coefs)

        reduced_matrix = torch.tensor(reduced_matrix, dtype=input_matrix.dtype, device=device)
        second_matrix = torch.tensor(svd.components_, dtype=input_matrix.dtype, device=device)

        svd_word_emb_module = MFWeightModule(latent_weights=reduced_matrix,
                                             second_matrix=second_matrix,
                                             module_bias=module_bias,
                                             original_params_num=original_params_num)
        return svd_word_emb_module
