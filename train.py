import os
import torch
import torch.nn as nn
from utils import CRITERION, BasicArgs, parse_args, set_seed
from eval import EvalArgs, eval_model
from dataset import StockDataset, stock_collate
from torch.utils.data import DataLoader
from model import ModelArgs, TransformerModel, save_model
import torch.optim as optim
from os.path import join as pjoin


class TrainArgs(ModelArgs, BasicArgs):
    num_epochs: int = 50000
    """Number of epochs"""
    learning_rate: float = 0.0001
    """Learning rate"""
    save_interval: int = 10_000
    """Interval to save model"""
    save_dir: str = "./models/model/"
    """Directory to save model checkpoints"""
    log_interval: int = 1_000
    """Interval to log loss"""
    loss_window: int = 100
    """Number of losses to average"""


criterion = nn.MSELoss()


def model_size(model: nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2


def create_eval_args(args: TrainArgs, model_path: str) -> EvalArgs:
    eval_args = EvalArgs()
    for arg, value in args.__dict__.items():
        if hasattr(eval_args, arg):
            setattr(eval_args, arg, value)
    eval_args.model_path = model_path
    return eval_args


def train_model(args: TrainArgs, model: TransformerModel, train_dataloader: DataLoader, test_dataloader: DataLoader):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    model.to(args.device)
    print("model size: {:.3f}MB".format(model_size(model)))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    recent_losses = []
    for epoch in range(args.num_epochs):
        for index, (inputs, targets, _, _) in enumerate(train_dataloader):
            inputs: list[torch.Tensor]
            targets: torch.Tensor

            inputs = [input.to(args.device) for input in inputs]
            targets = targets.to(args.device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs: torch.Tensor = model(inputs)
            loss: torch.Tensor = CRITERION(outputs, targets)

            recent_losses.append(loss.item())
            if len(recent_losses) > args.loss_window:
                recent_losses.pop(0)

            if index % args.log_interval == 0:
                print(f"Epoch {epoch}, loss: {sum(recent_losses) / len(recent_losses)}")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        if epoch % args.save_interval == 0 and epoch > 0:
            print(f"Saving model at epoch {epoch}")
            save_path = pjoin(args.save_dir, f"checkpoint_{epoch}.pt")
            
            save_model(model, args, save_path)
            eval_model(create_eval_args(args, save_path), model, test_dataloader)
            print("Model saved")

    print("Training complete")


def main():
    args: TrainArgs = parse_args(TrainArgs)
    set_seed(args.seed)

    # Create dataset and dataloader
    train_dataset = StockDataset(pjoin(args.data_dir, "train"))
    test_dataset = StockDataset(pjoin(args.data_dir, "test"))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=stock_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=stock_collate)

    # Create model
    args.input_sizes = train_dataset.input_sizes
    model = TransformerModel(args)

    # Train model
    train_model(args, model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
