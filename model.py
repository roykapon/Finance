import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_sizes, num_heads=4, num_layers=2, hidden_dim=256, output_dim=1):
        super().__init__()

        # Create an MLP for each data type
        self.mlps = nn.ModuleList([MLP(input_size) for input_size in input_sizes])

        # Class token embedding
        self.class_token = nn.Parameter(torch.zeros(1, 1, 64))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final MLP to predict future stock value
        self.final_mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, output_dim))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        # Encode each input using its respective MLP
        encoded_inputs = [mlp(x) for mlp, x in zip(self.mlps, inputs)]

        # Concatenate encoded inputs and add class token
        LEN, BS, LATENT = encoded_inputs[0].shape
        x = torch.cat([torch.tile(self.class_token, [1, BS, 1])] + encoded_inputs, dim=0)

        # Apply Transformer encoder
        x = self.transformer_encoder(x)

        # Extract class token and apply final MLP
        return self.final_mlp(x[0])