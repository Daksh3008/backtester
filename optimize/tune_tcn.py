# optuna/tune_tcn.py

import optuna
import json
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from optimize.utils.objective_helpers import train_eval_tcn

FEATURE_PATH = "data/feature_matrix.csv"
BEST_SAVE = "optimize/best_params/tcn_best.json"


def objective(trial: optuna.Trial):
    seq_len = trial.suggest_int("seq_len", 60, 180)
    channels = trial.suggest_int("channels", 32, 128)
    layers = trial.suggest_int("layers", 3, 7)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 20, 60)

    params = {
        "seq_len": seq_len,
        "channels": channels,
        "layers": layers,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    df = pd.read_csv(FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return train_eval_tcn(df, params)


def main():
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),   # <-- FIXED
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=10)

    print("Best TCN params:\n", study.best_trial.params)

    with open(BEST_SAVE, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)


if __name__ == "__main__":
    main()
