import os
import numpy as np
import torch
from dataset import DATA_TYPES, STOCK_PRICE_OPEN_INDEX, WINDOW_SIZE, StockDataset, stock_collate
from os.path import join as pjoin
from eval import EvalArgs
from model import TransformerModel, load_model
from torch.utils.data import DataLoader
from utils import CRITERION, parse_args, set_seed, visualize_stocks


class InferenceArgs(EvalArgs):
    pass


@torch.no_grad()
def main():
    args: InferenceArgs = parse_args(InferenceArgs)
    set_seed(args.seed)
    args.save_dir = args.save_dir if args.save_dir else pjoin(os.path.dirname(args.model_path))

    test_dataset = StockDataset(pjoin(args.data_dir, "test"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=stock_collate)
    model, model_args = load_model(args.model_path)
    model.to(args.device)
    model.eval()

    inputs, targets, companies, dates = next(iter(test_dataloader))
    inputs: list[torch.Tensor]
    targets: torch.Tensor
    companies: list[str]
    dates: np.ndarray

    inputs = [input.to(args.device) for input in inputs]
    targets = targets.to(args.device)

    outputs: torch.Tensor = model(inputs)
    loss: torch.Tensor = CRITERION(outputs, targets)

    print(f"Loss: {loss.item()}")
    # save inference results to a file
    input_stock_prices = inputs[DATA_TYPES.index("stock_prices")][..., STOCK_PRICE_OPEN_INDEX].cpu().numpy()
    predicted_stock_prices = np.concatenate((input_stock_prices, outputs.cpu().numpy()[np.newaxis]))
    gt_stock_prices = np.concatenate((input_stock_prices, targets.cpu().numpy()[np.newaxis]))
    dates = test_dataset.denormalize_dates(dates)
    dates_predicted = np.stack([dates + (i - input_stock_prices.shape[0]) for i in range(input_stock_prices.shape[0])] + [dates + WINDOW_SIZE])
    visualize_stocks(predicted_stock_prices.transpose(), gt_stock_prices.transpose(), companies, dates_predicted.transpose(), args.save_dir)
    print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
