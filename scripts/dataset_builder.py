# scripts/dataset_builder.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, X_df, y_series, seq_len=120):
        self.X = X_df.values
        self.y = y_series.values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
