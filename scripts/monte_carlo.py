# scripts/monte_carlo.py
"""
Monte Carlo simulations for recursive log-return forecasts.

Functions:
- monte_carlo_paths(mean_rets, logvar_rets, start_price, n_sims=200, random_state=None)
    -> returns np.ndarray of shape (n_sims, n_steps+1) with column 0 = start_price

- summarize_paths(paths, percentiles=(5,50,95)) -> dict of percentile arrays (each length n_steps+1)

- plot_simulation(dates, paths, actual_series=None, out_path=..., title=...)
    -> saves PNG to out_path. Uses matplotlib; does not set colors explicitly (per instructions).
"""

from __future__ import annotations
import os
from typing import Iterable, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def monte_carlo_paths(
    mean_rets: Iterable[float],
    logvar_rets: Iterable[float],
    start_price: float,
    n_sims: int = 200,
    random_state: int | None = None,
) -> np.ndarray:
    mean_arr = np.asarray(list(mean_rets), dtype=float)
    logvar_arr = np.asarray(list(logvar_rets), dtype=float)
    if mean_arr.shape != logvar_arr.shape:
        raise ValueError("mean_rets and logvar_rets must have same length")
    rng = np.random.default_rng(random_state)
    n_steps = len(mean_arr)
    paths = np.empty((n_sims, n_steps + 1), dtype=float)
    paths[:, 0] = start_price
    for t in range(n_steps):
        mu = mean_arr[t]
        var = float(np.exp(logvar_arr[t]))  # ensure positive
        sigma = np.sqrt(max(var, 1e-12))
        z = rng.standard_normal(size=n_sims)
        draws = mu + sigma * z  # log-returns draws
        paths[:, t + 1] = paths[:, t] * np.exp(draws)
    return paths


def summarize_paths(paths: np.ndarray, percentiles: Tuple[int, ...] = (5, 50, 95)) -> Dict[int, np.ndarray]:
    # paths shape (n_sims, n_steps+1)
    pct = {}
    for p in percentiles:
        pct[p] = np.percentile(paths, p, axis=0)
    return pct


def plot_simulation(
    dates: pd.DatetimeIndex,
    paths: np.ndarray,
    actual_series: pd.Series | None = None,
    out_path: str = "outputs/simulations/simulation.png",
    title: str | None = None,
    show_mean: bool = True,
    pct_bands: Tuple[int, int] = (5, 95),
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_sims, n_points = paths.shape
    if len(dates) != n_points:
        raise ValueError("dates length must equal number of columns in paths")
    plt.figure(figsize=(12, 6))
    # plot many simulations (light)
    for i in range(min(n_sims, 500)):
        plt.plot(dates, paths[i, :], linewidth=0.5, alpha=0.12)
    # percentiles
    pct = summarize_paths(paths, percentiles=(pct_bands[0], 50, pct_bands[1]))
    plt.plot(dates, pct[50], linewidth=2.0, label="MC mean/median")
    plt.fill_between(dates, pct[pct_bands[0]], pct[pct_bands[1]], alpha=0.18, label=f"MC {pct_bands[0]}-{pct_bands[1]} band")
    if actual_series is not None:
        # align
        a = actual_series.reindex(dates)
        plt.plot(dates, a.values, linewidth=2.0, linestyle="--", label="actual")
    plt.xlabel("Date")
    plt.ylabel("Price")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
