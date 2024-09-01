import os
import torch
from load_data import StockDataset, stock_collate
from os.path import join as pjoin
from model import TransformerModel
from torch.utils.data import DataLoader
from utils import CRITERION, parse_args


class EvalArgs:
    data_dir: str = "./data"
    """Path to data directory"""
    model_path: str = "./model/checkpoint_<epoch>.pt"
    """Path to model checkpoint"""
    batch_size: int = 4
    """Batch size for training"""
    device: str = "cuda:0"
    """cuda device index"""
    save_dir: str = "./model/"
    """Directory to save model checkpoints"""

@torch.no_grad()
def eval_model(args: EvalArgs, model: TransformerModel, test_dataloader: DataLoader):
    model.eval()
    losses = []
    for inputs, targets, dates in test_dataloader:
        inputs: list[torch.Tensor]
        targets: torch.Tensor
        dates: torch.Tensor

        inputs = [input.to(args.device) for input in inputs]
        targets = targets.to(args.device)
        
        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = CRITERION(outputs.squeeze(), targets)

        losses.append(loss.item())
    
    print(f"Average loss: {sum(losses) / len(losses)}")
    # save evaluation results to a file
    results_path = pjoin(os.path.dirname(args.model_path), f"eval_results_{os.path.splitext(os.path.basename(args.model_path))[0]}.txt")
    with open(results_path, "w") as f:
        f.write(f"Average loss: {sum(losses) / len(losses)}")



if __name__ == "__main__":
    args: EvalArgs = parse_args()
    data = StockDataset(pjoin(args.data_dir, "test"))
    test_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, collate_fn=stock_collate)
    model = TransformerModel(data.input_sizes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    eval_model(args, model, test_dataloader)
