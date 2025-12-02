# scripts/train_tcn.py

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

from scripts.dataset_builder import SeqDataset

CLOSE_COL = "brent_Close"


class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        padding = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, dilation=d, padding=padding),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, k, dilation=d, padding=padding),
            Chomp1d(padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class TCNHetero(LightningModule):
    def __init__(self, input_dim, channels=64, layers=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        mods = []
        in_ch = input_dim
        for i in range(layers):
            out_ch = channels
            mods.append(TCNBlock(in_ch, out_ch, k=3, d=2**i))
            in_ch = out_ch

        self.tcn = nn.Sequential(*mods)
        self.mu = nn.Linear(channels, 1)
        self.logvar = nn.Linear(channels, 1)
        self.loss_fn = nn.MSELoss()  # simple regression loss for mu

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.tcn(x)         # (B, C, T)
        h_last = h[:, :, -1]    # (B, C)
        mu = self.mu(h_last).squeeze(-1)       # (B,)
        logvar = self.logvar(h_last).squeeze(-1)  # (B,)
        return mu, logvar

    def training_step(self, batch, _):
        X, y = batch
        mu, _ = self(X)
        loss = self.loss_fn(mu, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        X, y = batch
        mu, _ = self(X)
        loss = self.loss_fn(mu, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def _extract_xy(train_df: pd.DataFrame):
    if "log_ret" not in train_df.columns:
        raise ValueError("Expected 'log_ret' column in train_df")
    y = train_df["log_ret"].astype(float)
    feature_cols = [c for c in train_df.columns if c != "log_ret"]
    X = train_df[feature_cols].astype(float)
    return X, y, feature_cols


def train_and_get_model(train_df: pd.DataFrame, params: dict):
    seq_len = params.get("seq_len", 120)
    batch_size = params.get("batch_size", 64)
    epochs = params.get("epochs", 40)
    channels = params.get("channels", 64)
    layers = params.get("layers", 5)
    lr = params.get("lr", 1e-3)
    save_path = params.get("save_path", "models/tcn")
    os.makedirs(save_path, exist_ok=True)

    X, y, feature_cols = _extract_xy(train_df)

    n_total = len(train_df)
    n_train = int(n_total * 0.9)
    if n_train <= seq_len + 10:
        raise ValueError(f"Not enough data for TCN training: n_total={n_total}, seq_len={seq_len}")

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val = X.iloc[n_train - seq_len:]
    y_val = y.iloc[n_train - seq_len:]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=feature_cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=feature_cols)

    train_ds = SeqDataset(X_train_s, y_train, seq_len=seq_len)
    val_ds = SeqDataset(X_val_s, y_val, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TCNHetero(
        input_dim=len(feature_cols),
        channels=channels,
        layers=layers,
        lr=lr,
    )

    ckpt = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    es = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=[ckpt, es],
        logger=False,
    )
    trainer.fit(model, train_loader, val_loader)

    best_model = TCNHetero.load_from_checkpoint(ckpt.best_model_path)

    bundle = {
        "model": best_model.eval(),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "close_col": CLOSE_COL,
    }
    return bundle


@torch.no_grad()
@torch.no_grad()
def predict_recursive(model_bundle: dict,
                      full_df: pd.DataFrame,
                      start_price=None,
                      predict_dates: pd.DatetimeIndex = None):
    """
    Recursive TCN forecast.

    Returns:
        (mean_logret_series, logvar_series)
    """
    import torch
    
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_cols = model_bundle["feature_cols"]
    seq_len = model_bundle["seq_len"]
    close_col = model_bundle["close_col"]

    # ------------------ DEVICE ------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    df = full_df.copy().sort_index()

    if predict_dates is None or len(predict_dates) == 0:
        raise ValueError("predict_dates must be non-empty")

    missing = predict_dates.difference(df.index)
    if len(missing) > 0:
        raise ValueError(f"predict_dates missing in df index: {missing[:5]}")

    # scaled features
    X_all = df[feature_cols].astype(float)
    X_all_s = pd.DataFrame(scaler.transform(X_all), index=X_all.index, columns=feature_cols)

    # find start point
    first_pred_date = predict_dates[0]
    hist_idx = X_all_s.index[X_all_s.index < first_pred_date]
    if len(hist_idx) < seq_len:
        raise ValueError("Not enough history before first predict date for TCN window")

    start_date = hist_idx[-1]

    if start_price is None:
        if close_col not in df.columns:
            raise ValueError(f"{close_col} not found in df for start_price.")
        last_price = float(df.loc[start_date, close_col])
    else:
        last_price = float(start_price)

    pos = X_all_s.index.get_loc(start_date)
    window_idx = X_all_s.index[pos - seq_len + 1: pos + 1]

    # ------------------ WINDOW INIT ------------------ #
    window = torch.tensor(
        X_all_s.loc[window_idx].values,
        dtype=torch.float32,
        device=device               # <<<< ensure GPU at creation
    ).unsqueeze(0)  # (1, T, F)

    close_idx = feature_cols.index(close_col)

    mu_list = []
    logvar_list = []
    price_list = []
    curr_price = last_price

    model.eval()

    # ------------------ RECURSIVE LOOP ------------------ #
    for _ in range(len(predict_dates)):
        mu, logvar = model(window)  # window already on GPU

        mu_val = float(mu.squeeze().detach().cpu().numpy())
        logvar_val = float(logvar.squeeze().detach().cpu().numpy())

        mu_list.append(mu_val)
        logvar_list.append(logvar_val)

        # reconstruct price incrementally
        curr_price = curr_price * np.exp(mu_val)
        price_list.append(curr_price)

        # ------- next window build -------- #
        last_feats_scaled = window[0, -1, :].detach().cpu().numpy().reshape(1, -1)
        last_feats = scaler.inverse_transform(last_feats_scaled)[0]
        last_feats[close_idx] = curr_price

        new_feats_scaled = scaler.transform(last_feats.reshape(1, -1))[0]

        new_row = torch.tensor(
            new_feats_scaled,
            dtype=torch.float32,
            device=device       # <<<< ensure GPU
        ).view(1, 1, -1)

        window = torch.cat([window[:, 1:, :], new_row], dim=1)

    # ------------------ OUTPUT ------------------ #
    mu_series = pd.Series(mu_list, index=predict_dates, name="tcn_mean_logret")
    logvar_series = pd.Series(logvar_list, index=predict_dates, name="tcn_logvar_logret")

    return mu_series, logvar_series

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
