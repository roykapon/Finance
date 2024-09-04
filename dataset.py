from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from os.path import join as pjoin

# Global variable for the number of days to average
DAYS_TO_AVERAGE = 5
DATA_TYPES = ["stock_prices", "income_statement", "balance_sheet", "cash_flow"]  # removed "corporate_actions"
DATE_INDEX = 0
WINDOW_SIZE = 365  # a years of data
WINDOW_INTERVAL = 5  # 5 days
STOCK_PRICE_OPEN_INDEX = 1


def stock_collate(batch):
    inputs, targets, dates = zip(*batch)

    inputs = [[torch.FloatTensor(input[data_type]) for input in inputs] for data_type in DATA_TYPES]
    padded = [pad_sequence(input_of_type) for input_of_type in inputs]
    return padded, torch.FloatTensor(targets), torch.FloatTensor(dates)


def int_to_date(date):
    epoch = datetime(1, 1, 1)  # January 1st, year 0
    target_date = epoch + timedelta(days=int(date))
    return target_date.strftime("%Y-%m-%d")


def retrieve_company_data(data: dict[str, dict[str, np.ndarray]], company_name: str, date: float) -> dict[str, np.ndarray]:
    company_data = data[company_name]
    stock_prices = company_data["stock_prices"]
    
    # Generate dictionary up to, but not including, the sampled date for each data type
    date_indices = {data_type: np.where(data_of_type[:, 0] < date)[0].max(initial=0) for data_type, data_of_type in company_data.items()}
    data_up_to_date = {data_type: data_of_type[: date_indices[data_type] + 1] for data_type, data_of_type in company_data.items()}
    
    # Calculate average stock value over the next WINDOW_SIZE days
    stocks_data_after = stock_prices[date_indices["stock_prices"] : date_indices["stock_prices"] + WINDOW_SIZE, STOCK_PRICE_OPEN_INDEX]

    return data_up_to_date, stocks_data_after.mean(), date

class StockDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir: str = data_dir
        self.data: dict[str, dict[str, np.ndarray]] = self.load_data()
        assert len(self.data) > 0, "No data found"
        self.input_sizes: list[str] = [next(iter(self.data.values()))[data_type].shape[1] for data_type in DATA_TYPES]

    def load_data(self) -> dict[str, dict[str, np.ndarray]]:
        data = {company_name: {data_type: np.load(f"{self.data_dir}/{company_name}/{data_type}.npy") for data_type in DATA_TYPES} for company_name in os.listdir(self.data_dir) if os.path.isdir(pjoin(self.data_dir, company_name))}
        for company, company_data in data.items():
            company_data["stock_prices"] = company_data["stock_prices"][::WINDOW_INTERVAL]
        return data

    def __len__(self):
        return len(self.data) * 100

    def __getitem__(self, idx):
        # Pick a random company
        company: str = random.choice(list(self.data.keys()))
        company_data = self.data[company]
        
        # Sample a random date within the available range of stock prices
        stock_prices = company_data["stock_prices"]
        sampled_date_index = random.randint(stock_prices.shape[0] // 2, stock_prices.shape[0] - 1)
        sampled_date = stock_prices[sampled_date_index, DATE_INDEX]
        
        return retrieve_company_data(self.data, sampled_date, sampled_date)


