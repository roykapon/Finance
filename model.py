import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import BasicArgs


class ModelArgs(BasicArgs):
    input_sizes: list[int] = []
    """List of input sizes for each data type"""
    num_heads: int = 8
    """Number of attention heads"""
    num_layers: int = 4
    """Number of transformer layers"""
    ff_hidden_dim: int = 128
    """Hidden dimension of the feedforward network"""
    latent_dim: int = 64
    """Dimension of the latent space"""


class MLP(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))


class TransformerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.positional_encoding = PositionalEncoding(args.latent_dim)
        # Create an MLP for each data type and one for the class embeddings
        self.mlps = nn.ModuleList([MLP(input_size, args.ff_hidden_dim, args.latent_dim) for input_size in args.input_sizes] + [MLP(1, args.ff_hidden_dim, args.latent_dim)])

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.latent_dim, nhead=args.num_heads, dim_feedforward=args.ff_hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)

        # Final MLP to predict future stock value
        self.final_mlp = MLP(args.latent_dim, args.ff_hidden_dim, 1)

    def forward(self, inputs: list[torch.Tensor], out_dates: torch.Tensor) -> torch.Tensor:
        # add class tokens
        # Encode each input using its respective MLP
        x: list[torch.Tensor] = torch.concatenate([mlp(x) for mlp, x in zip(self.mlps, inputs + [out_dates.unsqueeze(-1)])], axis=0)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply Transformer encoder
        x = self.transformer_encoder(x)

        # Extract class token and apply final MLP
        return self.final_mlp(x[-out_dates.shape[0] :])[:, :, 0]


def save_model(model: TransformerModel, model_args, path: str):
    torch.save({"model": model.state_dict(), "model_args": model_args}, path)


def load_model(path: str) -> tuple[TransformerModel, ModelArgs]:
    checkpoint = torch.load(path)
    model = TransformerModel(checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["model_args"]


class PositionalEncoding(nn.Module):

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Create a matrix of shape (max_len, d_model) to hold the positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add batch dimension

        # Register the positional encoding as a buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to the input
        x = x + self.pe[: x.shape[0], ...]
        return self.dropout(x)
