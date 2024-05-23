import torch
from wenet.transformer.embedding import PositionalEncoding, RelPositionalEncoding

class EmbeddingFrontend(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        positional_enc_type: str = "rel_pos",
        positional_dropout_rate: float = 0.1,
        padding_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        if positional_enc_type == "abs_pos":
            positional_enc = PositionalEncoding
        elif positional_enc_type == "rel_pos":
            positional_enc = RelPositionalEncoding
        else:
            positional_enc = None

        self.embed_dim = embed_dim
        if padding_idx is not None:
            input_size += 1  # for padding 0
        
        layers = [torch.nn.Embedding(input_size, embed_dim, padding_idx=padding_idx)]
        if positional_enc:
            layers.append(positional_enc(embed_dim, positional_dropout_rate))
        self.embed = torch.nn.Sequential(*layers)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ):
        x = self.embed(input)  # [B, T, *]
        output = x[0] if isinstance(x, tuple) else x
        return output, input_lengths