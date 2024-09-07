import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import BasicArgs


class ModelArgs(BasicArgs):
    input_sizes: list[int] = []
    """List of input sizes for each data type"""
    num_heads: int = 4
    """Number of attention heads"""
    num_layers: int = 2
    """Number of transformer layers"""
    ff_hidden_dim: int = 32
    """Hidden dimension of the feedforward network"""
    latent_dim: int = 16
    """Dimension of the latent space"""


class MLP(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))


class TransformerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Create an MLP for each data type
        self.mlps = nn.ModuleList([MLP(input_size, args.ff_hidden_dim, args.latent_dim) for input_size in args.input_sizes])

        # Class token embedding
        self.class_token = nn.Parameter(torch.zeros(1, 1, args.latent_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.latent_dim, nhead=args.num_heads, dim_feedforward=args.ff_hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)

        # Final MLP to predict future stock value
        self.final_mlp = MLP(args.latent_dim, args.ff_hidden_dim, 1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        # Encode each input using its respective MLP
        encoded_inputs = [mlp(x) for mlp, x in zip(self.mlps, inputs)]

        # Concatenate encoded inputs and add class token
        LEN, BS, LATENT = encoded_inputs[0].shape
        x = torch.cat([torch.tile(self.class_token, [1, BS, 1])] + encoded_inputs, dim=0)

        # Apply Transformer encoder
        x = self.transformer_encoder(x)

        # Extract class token and apply final MLP
        return self.final_mlp(x[0])[:, 0]


def save_model(model: TransformerModel, model_args, path: str):
    torch.save({"model": model.state_dict(), "model_args": model_args}, path)


def load_model(path: str) -> tuple[TransformerModel, ModelArgs]:
    checkpoint = torch.load(path)
    model = TransformerModel(checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["model_args"]
