import os
import torch.nn as nn
from argparse import ArgumentParser, Namespace
import inspect
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from dataset import STOCK_PRICE_OPEN_INDEX

CRITERION = nn.MSELoss()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_attr_doc(cls: type, attr: str):
    # get the source code of the attribute
    for base_cls in cls.__mro__:
        try:
            source = inspect.getsource(base_cls)
            lines = source.split("\n")
            attr_doc = ""
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{attr}"):
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith('"""'):
                        attr_doc = lines[i + 1].strip().strip('"""')
                        return f"{attr_doc} ({getattr(cls, attr)})"

            if base_cls != cls:
                get_attr_doc(base_cls, attr)

        except (TypeError, OSError) as e:
            continue  # Skip if we can't get the source of this base class


def parse_args(ArgsClass: type):
    parser = ArgumentParser()
    for arg, value in inspect.getmembers(ArgsClass):
        if not arg.startswith("__"):
            parser.add_argument(f"--{arg}", type=type(value), default=value, help=get_attr_doc(ArgsClass, arg))
    return parser.parse_args()


class BasicArgs:
    data_dir: str = "./data"
    """Path to data directory"""
    batch_size: int = 4
    """Batch size for training"""
    device: str = "cuda:0"
    """cuda device index"""
    seed: int = 0
    """Seed to use for randomness"""


def args_to_dict(args: Namespace):
    return {arg: value for arg, value in args.__dict__.items() if not arg.startswith("__")}


def dict_to_args(args_dict: dict, ArgsClass: type):
    args = ArgsClass()
    for arg, value in args_dict.items():
        setattr(args, arg, value)
    return args