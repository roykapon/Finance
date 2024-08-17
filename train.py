import torch
import torch.nn as nn
from load_data import StockDataset, stock_collate
from torch.utils.data import DataLoader
from model import TransformerModel
import torch.optim as optim
from argparse import ArgumentParser
from os.path import join as pjoin

class TrainArgs:
    data_dir: str
    """Path to data directory"""
    batch_size: int
    """Batch size for training"""
    num_epochs: int
    """Number of epochs"""
    learning_rate: float
    """Learning rate"""
    device: str
    """cuda device index"""


def parse_args() -> TrainArgs:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50000, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default="0", help="cuda device index")
    return parser.parse_args()


def train_model(args: TrainArgs, model: TransformerModel, dataloader: DataLoader):
    device = f"cuda:{args.device}"
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    model.to(device)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(args.num_epochs):
        for inputs, targets, dates in dataloader:
            inputs: list[torch.Tensor]; targets: torch.Tensor; dates: torch.Tensor
            inputs = [input.to(device) for input in inputs]
            targets = targets.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs: torch.Tensor = model(inputs)
            loss: torch.Tensor = criterion(outputs.squeeze(), targets)

            print(f"Loss: {loss.item()}")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    print("Training complete")


if __name__ == "__main__":
    company_name = "AAPL"  # Replace with actual company name
    args = parse_args()

    # Create dataset and dataloader
    dataset = StockDataset(pjoin(args.data_dir, company_name))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=stock_collate)

    # Create model
    model = TransformerModel(dataset.input_sizes)

    # Train model
    train_model(args, model, dataloader)
