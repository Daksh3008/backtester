# optimize/utils/objective_helpers.py

"""
Common helper functions for Optuna hyperparameter tuning
with your existing recursive models (LSTM, TCN, Ridge).

Each train_eval_*:
- trains the model on the FULL df with candidate params
- evaluates recursive predictions on a small tail window
- returns validation RMSE on price
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


CLOSE_COL = "brent_Close"


def _build_val_window(df: pd.DataFrame, seq_len: int, n_pred: int = 40):
    """
    Build a validation slice:
      - last (seq_len + n_pred) rows of df
      - first seq_len rows = history
      - last n_pred rows = prediction period

    Returns:
      val_df, hist_idx, pred_idx
    """
    df = df.sort_index()
    if len(df) < seq_len + n_pred:
        # Not enough data for this hyperparam; return None and caller can penalize
        return None, None, None

    val_df = df.iloc[-(seq_len + n_pred):]
    hist_idx = val_df.index[:seq_len]
    pred_idx = val_df.index[seq_len:]
    return val_df, hist_idx, pred_idx


# ------------------------------------------------------------------
# LSTM
# ------------------------------------------------------------------

def train_eval_lstm(train_df: pd.DataFrame, params: dict) -> float:
    """
    Train LSTM with candidate params and return validation RMSE on price.
    Uses recursive prediction on the tail window.
    """
    import scripts.train_lstm as LSTM  # local import to avoid circular issues

    seq_len = int(params.get("seq_len", 120))
    n_pred = 40  # days to evaluate on

    val_df, hist_idx, pred_idx = _build_val_window(train_df, seq_len, n_pred)
    if val_df is None:
        # penalize this configuration
        return 1e6

    # Train model on the full training df (as your pipeline does)
    bundle = LSTM.train_and_get_model(train_df, params)

    # Actual prices on prediction window
    actual = val_df.loc[pred_idx, CLOSE_COL].astype(float)

    # Recursive forecast over the validation slice
    pred = LSTM.predict_recursive(
        bundle,
        full_df=val_df,
        start_price=None,
        predict_dates=pred_idx,
    )

    # Align and compute RMSE
    common = actual.index.intersection(pred.index)
    if len(common) == 0:
        return 1e6

    rmse = np.sqrt(mean_squared_error(actual.loc[common], pred.loc[common]))
    return float(rmse)


# ------------------------------------------------------------------
# TCN
# ------------------------------------------------------------------

def train_eval_tcn(train_df: pd.DataFrame, params: dict) -> float:
    """
    Train TCN (heteroskedastic head) with candidate params and
    return validation RMSE on price, using recursive μ(log-returns).
    """
    import scripts.train_tcn as TCN

    seq_len = int(params.get("seq_len", 120))
    n_pred = 40

    val_df, hist_idx, pred_idx = _build_val_window(train_df, seq_len, n_pred)
    if val_df is None:
        return 1e6

    bundle = TCN.train_and_get_model(train_df, params)

    # Actual prices
    actual = val_df.loc[pred_idx, CLOSE_COL].astype(float)

    # Recursive TCN predicts (mu_series, logvar_series) on log-returns
    mu_series, logvar_series = TCN.predict_recursive(
        bundle,
        full_df=val_df,
        start_price=None,
        predict_dates=pred_idx,
    )

    # Reconstruct price path from μ log-returns
    # start from last price before first prediction date
    start_price = float(val_df.loc[hist_idx[-1], CLOSE_COL])
    mu_vals = mu_series.reindex(pred_idx).values.astype(float)

    prices = start_price * np.exp(np.cumsum(mu_vals))
    pred = pd.Series(prices, index=pred_idx)

    common = actual.index.intersection(pred.index)
    if len(common) == 0:
        return 1e6

    rmse = np.sqrt(mean_squared_error(actual.loc[common], pred.loc[common]))
    return float(rmse)


# ------------------------------------------------------------------
# Ridge
# ------------------------------------------------------------------

def train_eval_ridge(train_df: pd.DataFrame, params: dict) -> float:
    """
    Train Ridge recursive model and return validation RMSE on price.
    """
    import scripts.train_ridge as RIDGE

    seq_len = int(params.get("seq_len", 120))
    n_pred = 40

    val_df, hist_idx, pred_idx = _build_val_window(train_df, seq_len, n_pred)
    if val_df is None:
        return 1e6

    bundle = RIDGE.train_and_get_model(train_df, params)

    actual = val_df.loc[pred_idx, CLOSE_COL].astype(float)

    pred = RIDGE.predict_recursive(
        bundle,
        full_df=val_df,
        start_price=None,
        predict_dates=pred_idx,
    )

    common = actual.index.intersection(pred.index)
    if len(common) == 0:
        return 1e6

    rmse = np.sqrt(mean_squared_error(actual.loc[common], pred.loc[common]))
    return float(rmse)
