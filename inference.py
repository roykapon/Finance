import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from dataset import DATA_TYPES, STOCK_PRICE_OPEN_INDEX, WINDOW_SIZE, StockDataset, stock_collate
from os.path import join as pjoin
from eval import EvalArgs
from model import TransformerModel, load_model
from torch.utils.data import DataLoader
from utils import CRITERION, parse_args, set_seed


class InferenceArgs(EvalArgs):
    pass


def visualize_results(inputs: list[torch.Tensor], outputs: torch.Tensor, targets: torch.Tensor, companies: list[str], dates: np.ndarray, test_dataset: StockDataset, args: InferenceArgs):
    input_batch = inputs[DATA_TYPES.index("stock_prices")][..., STOCK_PRICE_OPEN_INDEX].cpu().numpy().transpose()
    output_batch = outputs.cpu().numpy().transpose()
    output_batch = 2 * output_batch - input_batch[:, -1]

    dates = test_dataset.denormalize_dates(dates)
    prediction_batch = np.stack((input_batch[:, -1], output_batch), axis=1)
    dates_of_prediction_batch = np.stack((dates, dates + WINDOW_SIZE), axis=1)

    gt_batch = [test_dataset.retrieve_company_data(company, date + WINDOW_SIZE)[0]["stock_prices"][..., STOCK_PRICE_OPEN_INDEX] for company, date in zip(companies, dates)]
    dates_of_gt_batch = [np.arange(start_date, end_date) for start_date, end_date in zip((dates + WINDOW_SIZE) - np.array([len(gt) for gt in gt_batch]), dates + WINDOW_SIZE)]

    os.makedirs(args.save_dir, exist_ok=True)
    for i, (prediction, dates_of_prediction, gt, dates_of_gt, company) in enumerate(zip(prediction_batch, dates_of_prediction_batch, gt_batch, dates_of_gt_batch, companies)):
        plt.title(f"Stock Prices of {company} in dates: {dates[0]} to {dates[-1]}")
        plt.plot(dates_of_prediction, prediction, label="Prediction", color="red")
        plt.plot(dates_of_gt, gt, label="Ground Truth", color="blue")
        plt.savefig(pjoin(args.save_dir, f"{i}.png"))
        plt.clf()

    print(f"Results saved to {args.save_dir}")


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
    visualize_results(inputs, outputs, targets, companies, dates, test_dataset, args)


if __name__ == "__main__":
    main()
