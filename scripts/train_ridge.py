# scripts/train_ridge.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Sequence

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TARGET_COL = "log_ret"


@dataclass
class RidgeBundle:
    model: Ridge
    scaler: StandardScaler
    feature_cols: Sequence[str]


def _build_xy(df: pd.DataFrame, feature_cols, target_col: str):
    """
    One-step-ahead setup:
    X_t = features at time t
    y_t = log_ret at time t+1

    So we shift the target by -1.
    """
    X = df[feature_cols].values
    y = df[target_col].shift(-1).values
    # drop last row (no y)
    X = X[:-1]
    y = y[:-1]
    return X, y


def train_and_get_model(train_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a simple Ridge regression on one-step-ahead log-returns.

    We keep things simple & stable:
    - fit on all available train_df rows
    - StandardScaler on features
    """
    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Expected '{TARGET_COL}' in train_df")

    # all non-target numeric cols as features
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    X_train, y_train = _build_xy(train_df, feature_cols, TARGET_COL)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    alpha = float(params.get("alpha", 1.0))
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    model.fit(Xs, y_train)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def predict_recursive(
    model_bundle: Dict[str, Any],
    full_df: pd.DataFrame,
    start_price: float | None,
    predict_dates: pd.DatetimeIndex,
):
    """
    For each predict date D, we predict log_ret(D) using features at D-1.

    We then reconstruct a price path starting from:
    - start_price if provided
    - else last known close before first predict date

    NOTE: This is NOT multi-step recursive on predicted prices.
    It is a 1-step-ahead model using real historical features,
    which keeps it more stable than deep recursive models.
    """
    model: Ridge = model_bundle["model"]
    scaler: StandardScaler = model_bundle["scaler"]
    feature_cols = model_bundle["feature_cols"]

    # ensure index is datetime
    full_df = full_df.copy()
    full_df.index = pd.to_datetime(full_df.index)

    # we will predict log_ret for each D in predict_dates, using features at D-1
    logrets = []
    for d in predict_dates:
        # find previous date in full_df index
        # we assume full_df is daily with no gaps outside weekends/holidays
        try:
            pos = full_df.index.get_loc(d)
        except KeyError:
            raise KeyError(f"Predict date {d} not found in full_df index")

        if pos == 0:
            raise ValueError(f"No previous row available before {d} for one-step Ridge forecast")

        prev_row = full_df.iloc[pos - 1]
        x = prev_row[feature_cols].values.reshape(1, -1)
        xs = scaler.transform(x)
        log_ret_pred = float(model.predict(xs)[0])
        logrets.append(log_ret_pred)

    logrets = np.array(logrets, dtype=float)

    # reconstruct price path
    if start_price is None:
        # last actual close before first predict date
        first_pos = full_df.index.get_loc(predict_dates[0])
        if first_pos == 0:
            raise ValueError("Cannot infer start_price: no prior close")
        start_price = float(full_df.iloc[first_pos - 1]["brent_Close"])

    prices = [start_price * np.exp(logrets[:i+1].sum()) for i in range(len(logrets))]
    price_series = pd.Series(prices, index=predict_dates, name="ridge_pred")

    return price_series
