import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from dataset import DATA_TYPES, DATE_INDEX, STOCK_PRICE_OPEN_INDEX, WINDOW_INTERVAL, WINDOW_SIZE, StockDataset, stock_collate
from os.path import join as pjoin
from eval import EvalArgs
from model import TransformerModel, load_model
from torch.utils.data import DataLoader
from utils import CRITERION, parse_args, set_seed


class InferenceArgs(EvalArgs):
    pass


def visualize_results(inputs: list[torch.Tensor], predictions: torch.Tensor, targets: torch.Tensor, target_dates: torch.Tensor, companies: list[str], test_dataset: StockDataset, args: InferenceArgs):
    input_batch = inputs[DATA_TYPES.index("stock_prices")][..., STOCK_PRICE_OPEN_INDEX].cpu().numpy().transpose()
    input_dates_batch = test_dataset.denormalize_dates(inputs[DATA_TYPES.index("stock_prices")][..., DATE_INDEX].cpu().numpy().transpose())

    prediction_batch = predictions.cpu().numpy().transpose()
    target_batch = targets.cpu().numpy().transpose()
    target_dates_batch = test_dataset.denormalize_dates(target_dates.cpu().numpy().transpose())

    os.makedirs(args.save_dir, exist_ok=True)
    for i, (inputs, input_dates, prediction, prediction_dates, targets, target_dates, company) in enumerate(zip(input_batch, input_dates_batch, prediction_batch, target_dates_batch, target_batch, target_dates_batch, companies)):
        plt.title(f"Prediction of stock prices of {company} in dates: {target_dates[0]} to {target_dates[-1]}")
        plt.plot(input_dates, inputs, label="Inputs", color="red")
        plt.plot(prediction_dates, prediction, label="Predictions", color="green")
        plt.plot(target_dates, targets, label="Ground Truth", color="blue")
        plt.legend()
        plt.savefig(pjoin(args.save_dir, f"{i}.png"))
        plt.clf()

    print(f"Results saved to {args.save_dir}")


@torch.no_grad()
def main():
    args: InferenceArgs = parse_args(InferenceArgs)
    set_seed(args.seed)
    args.save_dir = args.save_dir if args.save_dir else pjoin(os.path.dirname(args.model_path))

    test_dataset = StockDataset(args.data_dir, test=False)
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

    target_dates = targets[:, :, 0]
    targets = targets[:, :, 1]

    predictions: torch.Tensor = model(inputs, target_dates)
    loss: torch.Tensor = CRITERION(predictions, targets)

    print(f"Loss: {loss.item()}")
    # save inference results to a file
    visualize_results(inputs, predictions, targets, target_dates, companies, test_dataset, args)


if __name__ == "__main__":
    main()
