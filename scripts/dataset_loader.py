# scripts/dataset_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DatasetLoader:
    """
    Loads feature matrix, constructs feature set, computes log-returns target,
    splits train/val, and returns standardized feature matrices.

    Target = log returns of Brent close.
    """

    def __init__(
        self,
        feature_path="data/feature_matrix.csv",
        base_close_col="brent_Close",     # <-- updated here
        train_end="2023-12-31"
    ):
        self.feature_path = feature_path
        self.base_close_col = base_close_col
        self.train_end = pd.to_datetime(train_end)
        self.scaler = None

        self.df = None
        self.X = None
        self.y = None
        self.feature_cols = None
        self.target_col = None

    def load(self):
        # Load CSV
        df = pd.read_csv(self.feature_path)

        # Ensure first column represents Date index
        if df.columns[0].lower() not in ["date", "datetime", "timestamp"]:
            print(f"[WARN] Renaming first column '{df.columns[0]}' to 'Date'")
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        if self.base_close_col not in df.columns:
            raise ValueError(
                f"Base close column '{self.base_close_col}' not found in feature matrix.\n"
                f"Available columns: {list(df.columns)[:10]}..."
            )

        # Compute log returns target
        df["log_ret"] = np.log(df[self.base_close_col] / df[self.base_close_col].shift(1))

        df = df.iloc[1:].copy()

        target_col = "log_ret"

        # Identify raw OHLCV columns for Brent
        raw_cols = [
            self.base_close_col,
            self.base_close_col.replace("_Close", "_Open"),
            self.base_close_col.replace("_Close", "_High"),
            self.base_close_col.replace("_Close", "_Low"),
            self.base_close_col.replace("_Close", "_Volume"),
            target_col
        ]

        feature_cols = [c for c in df.columns if c not in raw_cols]

        X = df[feature_cols]
        y = df[target_col]

        self.df = df
        self.X = X
        self.y = y
        self.feature_cols = feature_cols
        self.target_col = target_col

        print(f"[INFO] Loaded feature matrix: shape={df.shape}")
        print(f"[INFO] Features={len(feature_cols)}, Target='{target_col}'")


    def train_val_split(self):
        if self.df is None:
            raise RuntimeError("Call load() before train_val_split().")

        train_mask = self.df.index <= self.train_end
        val_mask = ~train_mask

        X_train = self.X.loc[train_mask]
        X_val = self.X.loc[val_mask]

        y_train = self.y.loc[train_mask]
        y_val = self.y.loc[val_mask]

        print(f"[INFO] Train rows: {len(X_train)}, Val rows: {len(X_val)}")

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)

        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            index=X_train.index, columns=self.feature_cols
        )

        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            index=X_val.index, columns=self.feature_cols
        )

        return (
            X_train_scaled, y_train,
            X_val_scaled, y_val
        )

    def get_scaler(self):
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Run train_val_split() first.")
        return self.scaler
