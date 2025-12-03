# scripts/ensemble.py
"""
Ensemble utilities.

Expectations:
- preds_dict: dict[str -> pd.Series] each indexed by date and containing predicted prices (aligned to same dates)
- actuals: pd.Series indexed by same dates with actual prices

Provides:
- fit_nnls_weights(preds_dict, actuals) -> np.array weights
- predict_with_weights(preds_dict, weights) -> pd.Series
- save_predictions(df, out_path) -> writes csv (used by rolling_backtest)
"""

from __future__ import annotations
import os
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge


def _stack_predictions(preds_dict: Dict[str, pd.Series]) -> Tuple[np.ndarray, pd.Index, Sequence[str]]:
    keys = list(preds_dict.keys())
    if not keys:
        raise ValueError("preds_dict empty")
    # Align indexes (intersection)
    idx = preds_dict[keys[0]].index
    for k in keys[1:]:
        idx = idx.intersection(preds_dict[k].index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates between prediction series")
    mat = np.vstack([preds_dict[k].loc[idx].values for k in keys]).T  # shape (n_samples, n_models)
    return mat, idx, keys


def fit_nnls_weights(preds_dict: Dict[str, pd.Series], actuals: pd.Series) -> Dict[str, float]:
    """
    Fit non-negative least-squares weights (nnls) to combine model predictions to match actuals.

    Returns dict mapping model_key -> weight (weights sum to 1 unless all zero).
    """
    mat, idx, keys = _stack_predictions(preds_dict)
    y = actuals.loc[idx].values
    # Solve NNLS
    w, r = nnls(mat, y)
    if w.sum() > 0:
        w = w / w.sum()
    # return as dict
    return {k: float(w[i]) for i, k in enumerate(keys)}


def fit_ridge_weights(
    preds_dict: Dict[str, pd.Series],
    actuals: pd.Series,
    alpha: float = 1.0
) -> Dict[str, float]:
    """
    Fit non-negative ridge regression weights:
    minimize || y - Xw ||^2 + alpha * ||w||^2, w >= 0.

    Then normalize to sum to 1 (if possible).
    """
    mat, idx, keys = _stack_predictions(preds_dict)
    y = actuals.loc[idx].values

    # Ridge with non-negative coeffs
    reg = Ridge(alpha=alpha, fit_intercept=False, positive=True)
    reg.fit(mat, y)
    w = reg.coef_.astype(float)

    if w.sum() > 0:
        w = w / w.sum()

    return {keys[i]: float(w[i]) for i in range(len(keys))}


def combine_with_prior(base_w: Dict[str, float], prior_w: Dict[str, float]) -> Dict[str, float]:
    """
    Multiply base weights by regime prior weights and renormalize.
    """
    keys = set(base_w.keys()) | set(prior_w.keys())
    w = np.array([base_w.get(k, 0.0) * prior_w.get(k, 1.0) for k in keys], dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    return {k: float(w[i]) for i, k in enumerate(keys)}


def decayed_weight_memory(
    history: Sequence[Dict[str, float]],
    decay: float = 0.8,
    max_months: int = 12,
) -> Dict[str, float]:
    """
    Combine up to last `max_months` weight dicts with exponential decay.
    history: list of past month weight dicts in chronological order.
    """
    if not history:
        return {}

    used = history[-max_months:]
    keys = set().union(*used)
    accum = {k: 0.0 for k in keys}

    # newest has smallest age
    for age, wdict in enumerate(reversed(used)):
        factor = decay ** age
        for k, v in wdict.items():
            accum[k] += factor * v

    total = sum(accum.values())
    if total > 0:
        for k in accum:
            accum[k] /= total
    return accum


def predict_with_weights(preds_dict: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    mat, idx, keys = _stack_predictions(preds_dict)
    w_vec = np.array([weights.get(k, 0.0) for k in keys], dtype=float)
    preds = mat.dot(w_vec)
    return pd.Series(preds, index=idx, name="ensemble_pred")


def save_predictions(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=True)
