# scripts/train_lstm.py

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


# ----------------------- Model definition ----------------------- #

class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # h: (B, T, H)
        score = self.v(torch.tanh(self.W(h))).squeeze(-1)  # (B, T)
        weights = torch.softmax(score, dim=1)              # (B, T)
        ctx = torch.sum(h * weights.unsqueeze(-1), dim=1)  # (B, H)
        return ctx


class LSTMAttentionModel(LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attn = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)         # (B, T, H)
        ctx = self.attn(out)          # (B, H)
        return self.fc(ctx).squeeze(-1)  # (B,)

    def training_step(self, batch, _):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ----------------------- Training API ----------------------- #

def _extract_xy(train_df: pd.DataFrame):
    if "log_ret" not in train_df.columns:
        raise ValueError("Expected 'log_ret' column in train_df")

    y = train_df["log_ret"].astype(float)
    # use all other columns as features (including brent_Close)
    feature_cols = [c for c in train_df.columns if c != "log_ret"]
    X = train_df[feature_cols].astype(float)
    return X, y, feature_cols


def train_and_get_model(train_df: pd.DataFrame, params: dict):
    """
    Train LSTM+Attention on log returns.

    params:
        seq_len, batch_size, epochs, hidden_dim, num_layers, lr, save_path, device
    """
    seq_len = params.get("seq_len", 120)
    batch_size = params.get("batch_size", 64)
    epochs = params.get("epochs", 40)
    hidden_dim = params.get("hidden_dim", 128)
    num_layers = params.get("num_layers", 2)
    lr = params.get("lr", 1e-3)
    save_path = params.get("save_path", "models/lstm")
    os.makedirs(save_path, exist_ok=True)

    X, y, feature_cols = _extract_xy(train_df)

    # time-based train/val split (last 10% as val)
    n_total = len(train_df)
    n_train = int(n_total * 0.9)
    if n_train <= seq_len + 10:
        raise ValueError(f"Not enough data for LSTM training: n_total={n_total}, seq_len={seq_len}")

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val = X.iloc[n_train - seq_len:]  # keep overlap for windows
    y_val = y.iloc[n_train - seq_len:]

    # scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=feature_cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=feature_cols)

    train_ds = SeqDataset(X_train_s, y_train, seq_len=seq_len)
    val_ds = SeqDataset(X_val_s, y_val, seq_len=seq_len)

    # use more workers for performance
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=15,
        pin_memory=True,
        persistent_workers=True
    )

    model = LSTMAttentionModel(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
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

    best_model = LSTMAttentionModel.load_from_checkpoint(ckpt.best_model_path)

    bundle = {
        "model": best_model.eval(),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "close_col": CLOSE_COL,
    }
    return bundle


# ----------------------- Recursive prediction ----------------------- #

@torch.no_grad()
def predict_recursive(
    model_bundle: dict,
    full_df: pd.DataFrame,
    start_price=None,
    predict_dates: pd.DatetimeIndex = None
) -> pd.Series:
    """
    Recursive price forecasting with LSTM.

    model_bundle: returned by train_and_get_model
    full_df: full feature matrix (same structure as train_df), indexed by Date
    start_price: optional; if None, use actual brent_Close at last train date
    predict_dates: DatetimeIndex of future dates to predict
    """
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_cols = model_bundle["feature_cols"]
    seq_len = model_bundle["seq_len"]
    close_col = model_bundle["close_col"]

    df = full_df.copy()
    df = df.sort_index()

    if predict_dates is None or len(predict_dates) == 0:
        raise ValueError("predict_dates must be a non-empty DatetimeIndex")

    # ensure we have all predict_dates in df
    missing = predict_dates.difference(df.index)
    if len(missing) > 0:
        raise ValueError(f"Some predict_dates not in df index: {missing[:5]}")

    # scaled features for all history
    X_all = df[feature_cols].astype(float)
    X_all_s = pd.DataFrame(
        scaler.transform(X_all),
        index=X_all.index,
        columns=feature_cols
    )

    # start date = last available date before first predict date
    first_pred_date = predict_dates[0]
    hist_idx = X_all_s.index[X_all_s.index < first_pred_date]
    if len(hist_idx) < seq_len:
        raise ValueError("Not enough history to build initial LSTM window before first predict date.")
    start_date = hist_idx[-1]

    # determine last_price
    if start_price is None:
        if close_col not in df.columns:
            raise ValueError(f"{close_col} not found in df for start_price.")
        last_price = float(df.loc[start_date, close_col])
    else:
        last_price = float(start_price)

    # build initial window (1, seq_len, F)
    pos = X_all_s.index.get_loc(start_date)
    window_idx = X_all_s.index[pos - seq_len + 1: pos + 1]
    window = torch.tensor(
        X_all_s.loc[window_idx].values,
        dtype=torch.float32
    ).unsqueeze(0)  # (1, seq_len, F)

    # device alignment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    window = window.to(device)

    # index of close price in feature vector
    close_idx = feature_cols.index(close_col)

    preds = []
    curr_price = last_price

    model.eval()

    for _ in range(len(predict_dates)):
        # predict next log return
        log_ret_pred = float(
            model(window)
            .detach()
            .cpu()
            .numpy()
            .ravel()[0]
        )

        # update price
        curr_price = curr_price * np.exp(log_ret_pred)
        preds.append(curr_price)

        # build next window
        # 1) inverse scale last row
        last_feats_scaled = window[0, -1, :].detach().cpu().numpy().reshape(1, -1)
        last_feats = scaler.inverse_transform(last_feats_scaled)[0]
        # 2) update close price
        last_feats[close_idx] = curr_price
        # 3) rescale
        new_feats_scaled = scaler.transform(last_feats.reshape(1, -1))[0]
        new_row = torch.tensor(
            new_feats_scaled,
            dtype=torch.float32
        ).view(1, 1, -1).to(device)

        # 4) roll window
        window = torch.cat([window[:, 1:, :], new_row], dim=1)

    pred_series = pd.Series(preds, index=predict_dates, name="lstm_pred")
    return pred_series

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
