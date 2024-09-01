import torch.nn as nn
from argparse import ArgumentParser

CRITERION = nn.MSELoss()


def parse_args(ArgsClass: type):
    parser = ArgumentParser()
    for arg, default in ArgsClass.__annotations__.items():
        parser.add_argument(f"--{arg}", type=type(ArgsClass.__dict__[arg]), default=ArgsClass.__dict__[arg], help=ArgsClass.__annotations__[arg])
    return parser.parse_args()
