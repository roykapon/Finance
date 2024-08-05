import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_sizes, num_heads=4, num_layers=2, hidden_dim=256, output_dim=1):
        super(TransformerModel, self).__init__()

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
        encoded_inputs = []
        for mlp, x in zip(self.mlps, inputs):
            mask = torch.isnan(x)
            x = x.masked_fill(mask, 0)  # Replace NaNs with zero
            encoded_inputs.append(mlp(x))

        # Concatenate encoded inputs and add class token
        LEN, BS, LATENT = encoded_inputs[0].shape
        x = torch.cat([torch.tile(self.class_token, [1, BS, 1])] + encoded_inputs, dim=0)

        # Apply Transformer encoder
        x = self.transformer_encoder(x)

        # Extract class token and apply final MLP
        return self.final_mlp(x[0])


# Example usage
if __name__ == "__main__":
    # Assuming each data type has different input sizes
    input_sizes = [10, 15, 20]  # Example input sizes for different data types

    # Create random input tensors for each data type
    inputs = [torch.rand(32, size) for size in input_sizes]  # Batch size of 32

    # Introduce some NaN values
    inputs[0][0, 0] = float("nan")
    inputs[1][1, 1] = float("nan")

    model = TransformerModel(input_sizes)
    output = model(inputs)
    print(output.shape)  # Should output: torch.Size([32, 1])
