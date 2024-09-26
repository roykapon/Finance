import os
from typing import Callable
import torch
from dataset import STOCK_PRICE_OPEN_INDEX, StockDataset, stock_collate
from os.path import join as pjoin
from model import TransformerModel, load_model
from torch.utils.data import DataLoader
from utils import get_loss_fn, parse_args, BasicArgs, set_seed


class EvalArgs(BasicArgs):
    model_path: str = ""
    """Path to model checkpoint"""
    save_dir: str = ""
    """Directory to save results"""


@torch.no_grad()
def eval_model(args: EvalArgs, model: TransformerModel, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], test_dataloader: DataLoader):
    model.eval()
    losses = []
    for inputs, targets, _, _ in test_dataloader:
        inputs: list[torch.Tensor]
        targets: torch.Tensor

        inputs = [input.to(args.device) for input in inputs]
        targets = targets.to(args.device)

        target_dates = targets[:, :, 0]
        targets = targets[:, :, 1]

        outputs: torch.Tensor = model(inputs, target_dates)
        loss: torch.Tensor = loss_fn(outputs, targets)

        losses.append(loss.item())

    print(f"Model: {args.model_path}    Loss: {sum(losses) / len(losses)}")
    # save evaluation results to a file
    results_path = pjoin(os.path.dirname(args.model_path), f"eval_results_{os.path.splitext(os.path.basename(args.model_path))[0]}.txt")
    with open(results_path, "w") as f:
        f.write(f"Average loss: {sum(losses) / len(losses)}")


def main():
    args: EvalArgs = parse_args(EvalArgs)
    set_seed(args.seed)
    data = StockDataset(pjoin(args.data_dir, test=True))
    test_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, collate_fn=stock_collate)
    model, model_args = load_model(args.model_path)
    model.to(args.device)
    loss_fn = get_loss_fn(data.mean["stock_prices"][STOCK_PRICE_OPEN_INDEX], data.std["stock_prices"][STOCK_PRICE_OPEN_INDEX])
    eval_model(args, model, loss_fn, test_dataloader)


if __name__ == "__main__":
    main()
