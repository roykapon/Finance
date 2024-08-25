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
    save_interval: int
    """Interval to save model"""


def parse_args() -> TrainArgs:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50000, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default="0", help="cuda device index")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval to save model")
    return parser.parse_args()


def model_size(model: nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2
    

def train_model(args: TrainArgs, model: TransformerModel, dataloader: DataLoader):
    device = f"cuda:{args.device}"
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    print('model size: {:.3f}MB'.format(model_size(model)))
    model.to(device)
    torch.autograd.set_detect_anomaly(True)

    recent_losses = []
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
            
            recent_losses.append(loss.item())
            if len(recent_losses) > 100:
                recent_losses.pop(0)
            print(f"Epoch {epoch}, loss: {sum(recent_losses) / len(recent_losses)}")~
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), pjoin(args.data_dir, "model.pth"))
                print("Model saved")

    print("Training complete")


if __name__ == "__main__":
    args = parse_args()

    # Create dataset and dataloader
    dataset = StockDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=stock_collate)

    # Create model
    model = TransformerModel(dataset.input_sizes)

    # Train model
    train_model(args, model, dataloader)
