# config/configs.py

CONFIG = {
    "seq_len": 120,
    "batch_size": 64,
    "epochs": 30,
    "lr": 1e-3,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "tcn_channels": [64, 64, 64],
    "dropout": 0.1,
    "xgb_params": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "tree_method": "hist"
    },
    "ensemble_val_window": 30,
    "mc_samples": 200,
}
