# optuna/tune_ridge.py

import optuna
import json
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from optimize.utils.objective_helpers import train_eval_ridge

FEATURE_PATH = "data/feature_matrix.csv"
BEST_SAVE = "optimize/best_params/ridge_best.json"


def objective(trial: optuna.Trial):
    seq_len = trial.suggest_int("seq_len", 60, 180)
    alpha = trial.suggest_float("alpha", 0.05, 4.0, log=True)

    params = {
        "seq_len": seq_len,
        "alpha": alpha,
    }

    df = pd.read_csv(FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return train_eval_ridge(df, params)


def main():
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),   # <-- FIXED
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=40)

    print("Best Ridge params:\n", study.best_trial.params)

    with open(BEST_SAVE, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)


if __name__ == "__main__":
    main()
