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
ABS_WINDOW_SIZE = 365  # a years of data
WINDOW_INTERVAL = 12  # 5 days
WINDOW_SIZE = ABS_WINDOW_SIZE // WINDOW_INTERVAL

STOCK_PRICE_OPEN_INDEX = 1


def stock_collate(batch: tuple[list[list[np.ndarray]], list[np.ndarray], list[str], list[float]]) -> tuple[list[np.ndarray], np.ndarray, list[str], np.ndarray]:
    inputs, targets, companies, dates = zip(*batch)

    inputs = [pad_sequence([torch.from_numpy(input[data_type].astype(np.float32)) for input in inputs]) for data_type in DATA_TYPES]
    targets = pad_sequence([torch.FloatTensor(target) for target in targets])
    return inputs, targets, companies, np.array(dates)


def int_to_date(date):
    epoch = datetime(1, 1, 1)  # January 1st, year 0
    target_date = epoch + timedelta(days=int(date))
    return target_date.strftime("%Y-%m-%d")


class StockDataset(Dataset):
    def __init__(self, data_dir: str, test: bool = False):
        self.data_dir: str = data_dir
        self.data: dict[str, dict[str, np.ndarray]] = self.load_data()
        self.test = test
        assert len(self.data) > 0, "No data found"
        self.input_sizes: list[str] = [next(iter(self.data.values()))[data_type].shape[1] for data_type in DATA_TYPES]
        self.mean: dict[str, np.ndarray] = np.load(pjoin(data_dir, "means.npy"), allow_pickle=True)[()]
        self.std: dict[str, np.ndarray] = np.load(pjoin(data_dir, "stds.npy"), allow_pickle=True)[()]

    def filter_data(self, data: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
        return {company: company_data for company, company_data in data.items() if company_data["stock_prices"].shape[0] >= 3 * ABS_WINDOW_SIZE}

    def load_data(self) -> dict[str, dict[str, np.ndarray]]:
        data: dict[str, dict[str, np.ndarray]] = {company_name: {data_type: np.load(pjoin(self.data_dir, company_name, f"{data_type}.npy")) for data_type in DATA_TYPES} for company_name in os.listdir(self.data_dir) if os.path.isdir(pjoin(self.data_dir, company_name))}
        data = self.filter_data(data)
        for company, company_data in data.items():
            company_data["stock_prices"] = company_data["stock_prices"][::WINDOW_INTERVAL]
        return data

    def __len__(self):
        return len(self.data) * 100

    def retrieve_company_data(self, company_name: str, date_from: float, date_to: float) -> dict[str, np.ndarray]:
        company_data = self.data[company_name]
        return {data_type: data_of_type[(data_of_type[:, 0] >= date_from) & (data_of_type[:, 0] <= date_to)] for data_type, data_of_type in company_data.items()}

    def sample_date(self, company_datas: np.ndarray) -> float:
        dates_len = company_datas.shape[0] - 1
        if self.test:
            date_index = random.randint(dates_len - 2 * WINDOW_SIZE, dates_len - WINDOW_SIZE)
        else:
            date_index = random.randint(2 * WINDOW_SIZE, dates_len - 2 * WINDOW_SIZE)
        return company_datas[date_index - 2 * WINDOW_SIZE], company_datas[date_index], company_datas[date_index + WINDOW_SIZE]

    def __getitem__(self, idx) -> tuple[list[np.ndarray], np.ndarray, str, float]:
        # Pick a random company
        company: str = random.choice(list(self.data.keys()))
        company_data = self.data[company]

        # Sample a random date within the available range of stock prices
        stock_prices = company_data["stock_prices"]
        input_start_date, date, prediction_end_date = self.sample_date(stock_prices[:, DATE_INDEX])

        past_data = self.retrieve_company_data(company, input_start_date, date)
        future_data = self.retrieve_company_data(company, input_start_date, prediction_end_date)
        future_stock_data = future_data["stock_prices"][:, (DATE_INDEX, STOCK_PRICE_OPEN_INDEX)]
        return past_data, future_stock_data, company, date

    def normalize_dates(self, dates: np.ndarray) -> np.ndarray:
        return (dates - self.mean["stock_prices"][DATE_INDEX]) / self.std["stock_prices"][DATE_INDEX]

    def denormalize_dates(self, dates: np.ndarray) -> np.ndarray:
        return dates * self.std["stock_prices"][DATE_INDEX] + self.mean["stock_prices"][DATE_INDEX]
