# scripts/rolling_backtest.py

from __future__ import annotations
import os
import numpy as np
import pandas as pd

# model modules
import scripts.train_lstm as LSTM
import scripts.train_tcn as TCN


from scripts import ensemble as ensemble_mod
from scripts import monte_carlo as mc_mod
from utils.io import save_excel


# ========================= Helpers ========================= #

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def last_day_of_month(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.month == 12:
        return pd.Timestamp(ts.year, 12, 31)
    return pd.Timestamp(ts.year, ts.month + 1, 1) - pd.Timedelta(days=1)


def month_range(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1)
    endm = pd.Timestamp(end.year, end.month, 1)
    while cur <= endm:
        yield cur
        if cur.month == 12:
            cur = pd.Timestamp(cur.year + 1, 1, 1)
        else:
            cur = pd.Timestamp(cur.year, cur.month + 1, 1)


# =============== Build monthly prediction table =============== #

def build_output_df(name, pred_s, actual_s):
    df = pd.DataFrame({
        "date": pred_s.index,
        "actual_price": actual_s.reindex(pred_s.index).values,
        "predicted_price": pred_s.values
    })
    df["pct_diff"] = (df["predicted_price"] - df["actual_price"]) / df["actual_price"] * 100
    df["model"] = name
    return df


def compute_rmse(actual, pred):
    idx = actual.index.intersection(pred.index)
    a = actual.loc[idx]
    p = pred.loc[idx]
    return float(np.sqrt(np.mean((a - p) ** 2)))


# ========================= Main Runner ========================= #

def run_backtest(
    feature_path="data/feature_matrix.csv",
    outputs_excel="outputs/backtest_excels",
    outputs_sims="outputs/simulations",
    start_train_cutoff="2023-12-31",
    final_predict_month="2025-11",
    seq_len=120,
    mc_sims=200
):
    ensure_dir(outputs_excel)
    ensure_dir(outputs_sims)

    # ---------------- load data ---------------- #
    df = pd.read_csv(feature_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.name = "Date"

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    if "log_ret" not in df.columns:
        raise ValueError("feature_matrix must contain 'log_ret'")

    if "brent_Close" not in df.columns:
        raise ValueError("feature_matrix must contain 'brent_Close'")

    CLOSE = "brent_Close"
    TARGET = "log_ret"

    # monthly iteration start and end
    start_m = pd.Timestamp(start_train_cutoff) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
    end_m = pd.Timestamp(final_predict_month + "-01")

    metrics_rows = []
    previous_weights = None  # <-- IMPORTANT

    for month in month_range(start_m, end_m):
        print(f"\n========== MONTH {month.strftime('%Y-%m')} ==========")

        train_end = (month - pd.Timedelta(days=1)).normalize()
        train_df = df.loc[df.index <= train_end]

        if len(train_df) < seq_len + 200:
            raise ValueError(f"Not enough training rows before {train_end}")

        p_start = pd.Timestamp(month.year, month.month, 1)
        p_end = last_day_of_month(p_start)

        predict_idx = df.index[(df.index >= p_start) & (df.index <= p_end)]
        if len(predict_idx) == 0:
            print("[WARN] No rows for this month, skipping")
            continue

        print(f"[INFO] train <= {train_end.date()} | predict {p_start.date()} -> {p_end.date()}")

        # ---------------- train models ---------------- #
        lstm_bundle = LSTM.train_and_get_model(train_df, {"seq_len": seq_len})
        tcn_bundle = TCN.train_and_get_model(train_df, {"seq_len": seq_len})

        # ---------------- predictions ---------------- #
        actuals = df.loc[predict_idx, CLOSE].astype(float)

        lstm_pred = LSTM.predict_recursive(lstm_bundle, df, start_price=None, predict_dates=predict_idx)
        lstm_pred.name = "lstm_pred"


        tcn_mu, tcn_logvar = TCN.predict_recursive(tcn_bundle, df, start_price=None, predict_dates=predict_idx)
        tcn_pred = (
            df.loc[predict_idx[0], CLOSE] * np.exp(np.cumsum(tcn_mu.values))
        )
        tcn_pred = pd.Series(tcn_pred, index=predict_idx, name="tcn_pred")

        preds_dict = {
            "lstm": lstm_pred,
            "tcn": tcn_pred,
        }

        # ---------------- ensemble ---------------- #
        if previous_weights is None:
            n_models = len(preds_dict)
            weights_to_apply = {k: 1.0 / n_models for k in preds_dict.keys()}
            print(f"[INFO] No historical weights -> using equal weights: {weights_to_apply}")
        else:
            weights_to_apply = previous_weights
            print(f"[INFO] Using previous month weights: {weights_to_apply}")

        ensemble_pred = ensemble_mod.predict_with_weights(preds_dict, weights_to_apply)
        ensemble_pred.name = "ensemble_pred"

        # ---------------- Fit NEW weights (using real prices) ---------------- #
        try:
            new_weights = ensemble_mod.fit_nnls_weights(preds_dict, actuals)
            previous_weights = new_weights  # used in NEXT iteration
            print(f"[INFO] NNLS fitted weights for next month: {new_weights}")
        except Exception as e:
            print("[WARN] NNLS weight fitting failed. Keeping previous weights.", e)

        # ---------------- Monte Carlo ---------------- #
        start_price = float(df.loc[df.index < predict_idx[0], CLOSE].iloc[-1])

        mc_paths = mc_mod.monte_carlo_paths(
            mean_rets=tcn_mu.values,
            logvar_rets=tcn_logvar.values,
            start_price=start_price,
            n_sims=mc_sims,
            random_state=42
        )

        mc_dates = list(predict_idx.insert(0, predict_idx[0] - pd.Timedelta(days=1)))
        mc_dates = pd.to_datetime(mc_dates)

        sim_plot = os.path.join(outputs_sims, f"simulation_{month.strftime('%Y_%m')}.png")
        mc_mod.plot_simulation(
            dates=mc_dates,
            paths=mc_paths,
            actual_series=None,
            out_path=sim_plot,
            title=f"MC {month.strftime('%Y-%m')}"
        )

        # ---------------- Excel Output ---------------- #
        excel_month = month.strftime("%Y_%m")
        excel_path = os.path.join(outputs_excel, f"{excel_month}.xlsx")

        dfs_excel = {
            "lstm": build_output_df("lstm", lstm_pred, actuals),
            "tcn": build_output_df("tcn", tcn_pred, actuals),
            "ensemble": build_output_df("ensemble", ensemble_pred, actuals),
        }

        mets = {
            "model": ["lstm", "tcn", "ensemble"],
            "rmse": [
                compute_rmse(actuals, lstm_pred),
                compute_rmse(actuals, tcn_pred),
                compute_rmse(actuals, ensemble_pred),
            ]
        }
        dfs_excel["metrics"] = pd.DataFrame(mets)

        save_excel(excel_path, dfs_excel)
        print(f"[SAVED] {excel_path}")

        # log monthly metrics
        for model_name, series in preds_dict.items():
            metrics_rows.append({
                "month": excel_month,
                "model": model_name,
                "rmse": compute_rmse(actuals, series),
            })

        metrics_rows.append({
            "month": excel_month,
            "model": "ensemble",
            "rmse": compute_rmse(actuals, ensemble_pred),
        })

    summary = pd.DataFrame(metrics_rows)
    summary.to_csv(os.path.join(outputs_excel, "metrics_summary.csv"), index=False)
    print("[DONE] Backtest complete, metrics_summary saved.")


# ---------------------- CLI Entrypoint ---------------------- #
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_backtest()
