from typing import Tuple

import torch
import torch.nn as nn
from tltorch.utils import FactorList
import tensorly as tl

tl.set_backend("pytorch")
from tensorly.decomposition import tucker

class TuckerDecompositionModel(nn.Module):
    def __init__(self, tucker_rank: list, input_matrix: torch.tensor, module_name: str,
                 num_layers: int, num_heads: int):
        super().__init__()

        if module_name in ["key", "query", "value"]:
            matrix_shape = input_matrix.shape
            self.hidden_dim = matrix_shape[1]
            if matrix_shape[0] == matrix_shape[1]:  # separated mode
                tensorized_input = input_matrix.reshape(num_heads, -1, self.hidden_dim)
            else:  # concatenated mode
                tensorized_input = input_matrix.reshape(num_layers * num_heads, -1, self.hidden_dim)
        else:  # word_embeddings
            assert module_name == "word_embeddings"

            vocab_size, self.hidden_dim = input_matrix.shape
            # adding padding to end of weights matrix to make it multiple of hidden_dim
            num_pad_rows = self.hidden_dim - vocab_size % self.hidden_dim
            tensorized_input = torch.cat([input_matrix,
                                          torch.zeros(num_pad_rows, self.hidden_dim)]).reshape(-1, self.hidden_dim,
                                                                                               self.hidden_dim)

        tensorized_input = tl.tensor(tensorized_input)

        self.rank = tucker_rank
        self.core, self.factors = tucker(tensorized_input, rank=self.rank, svd="randomized_svd")
        self.core = nn.Parameter(self.core)
        self.factors = FactorList([nn.Parameter(factor) for factor in self.factors])

    def forward(self, x, ids) -> Tuple[torch.Tensor, None]:
        res = tl.tucker_to_tensor((self.core, self.factors))

        return res.view(-1, self.hidden_dim)[ids, :], None

    def get_substitution_module_size(self) -> int:
        return self.core.numel() + sum([module_.numel() for module_ in self.factors])



