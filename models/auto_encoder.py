import torch
import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, weights_path: str = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def encode(self, input: torch.tensor, ids: torch.tensor) -> torch.tensor:
        latents = self.encoder(input)
        return latents

    def forward(self, input: torch.tensor, ids: torch.tensor) -> torch.tensor:
        latents = self.encode(input, ids)
        cloned_latents = latents.clone()
        out = self.decoder(latents, ids)
        return out, cloned_latents

    def get_substitution_module_size(self) -> int:
        return self.decoder.get_decoder_size()

    def get_latent_dim(self) -> int:
        return self.encoder.get_latent_dim()
