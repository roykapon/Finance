import os
from typing import Callable
import torch
import torch.nn as nn
from utils import BasicArgs, parse_args, set_seed, get_loss_fn
from eval import EvalArgs, eval_model
from dataset import DATE_INDEX, STOCK_PRICE_OPEN_INDEX, StockDataset, stock_collate
from torch.utils.data import DataLoader
from model import ModelArgs, TransformerModel, save_model
import torch.optim as optim
from os.path import join as pjoin


class TrainArgs(ModelArgs, BasicArgs):
    num_epochs: int = 50000
    """Number of epochs"""
    learning_rate: float = 0.0001
    """Learning rate"""
    save_interval: int = 100
    """Interval to save model"""
    eval_interval: int = 100
    """Interval to evaluate model"""
    save_dir: str = "./models/model/"
    """Directory to save model checkpoints"""
    log_interval: int = 1_000
    """Interval to log loss"""
    loss_window: int = 100
    """Number of losses to average"""


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


def train_model(args: TrainArgs, model: TransformerModel, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], train_dataloader: DataLoader, test_dataloader: DataLoader):
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
            """[stock_prices (LEN, BS, 8), income_statement (LEN2, BS, 25), balance_sheet (LEN3, BS, 37): , cash_flow (LEN, BS, 28)]"""
            targets: torch.Tensor

            inputs = [input.to(args.device) for input in inputs]
            targets = targets.to(args.device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            target_dates = targets[:, :, 0]
            targets = targets[:, :, 1]
            outputs: torch.Tensor = model(inputs, target_dates)
            loss: torch.Tensor = loss_fn(outputs, targets)

            recent_losses.append(loss.item())
            if len(recent_losses) > args.loss_window:
                recent_losses.pop(0)

            if index % args.log_interval == 0:
                print(f"Epoch {epoch}, loss: {sum(recent_losses) / len(recent_losses)}")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        if epoch % args.eval_interval == 0 and epoch > 0:
            eval_model(create_eval_args(args, pjoin(args.save_dir, f"checkpoint_eval_{epoch}.txt")), model, loss_fn, test_dataloader)

        if epoch % args.save_interval == 0 and epoch > 0:
            print(f"Saving model at epoch {epoch}")
            save_model(model, args, pjoin(args.save_dir, f"checkpoint_{epoch}.pt"))

    print("Training complete")


def main():
    args: TrainArgs = parse_args(TrainArgs)
    set_seed(args.seed)

    # Create dataset and dataloader
    train_dataset = StockDataset(args.data_dir, test=False)
    test_dataset = StockDataset(args.data_dir, test=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=stock_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=stock_collate)

    # Create model
    args.input_sizes = train_dataset.input_sizes
    model = TransformerModel(args)

    loss_fn = get_loss_fn(train_dataset.mean["stock_prices"][STOCK_PRICE_OPEN_INDEX], train_dataset.std["stock_prices"][STOCK_PRICE_OPEN_INDEX])

    # Train model
    train_model(args, model, loss_fn, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
