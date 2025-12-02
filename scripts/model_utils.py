# utils/model_utils.py

import torch
import numpy as np

def recursive_price_forecast(
    model,
    last_window_tensor,
    scaler,
    last_close_price,
    prediction_dates
):
    """
    model: trained PLModule
    last_window_tensor: shape (1, seq_len, feature_dim)
    scaler: sklearn scaler used for features
    last_close_price: float
    prediction_dates: list of future dates

    Model outputs log-return prediction.
    We accumulate to reconstruct price.
    """

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    preds = []
    curr_price = last_close_price
    window = last_window_tensor.clone().detach()

    for _ in range(len(prediction_dates)):

        with torch.no_grad():
            log_ret = model(window).cpu().numpy().ravel()[0]

        preds.append(log_ret)

        curr_price = curr_price * np.exp(log_ret)

        # update window
        # reconstruct features for next step = inverse transform previous window last row
        prev_feats = scaler.inverse_transform(window[0, -1, :].cpu().numpy().reshape(1, -1))[0]

        # update closing price feature with predicted price
        prev_feats[0] = curr_price

        new_feats_scaled = scaler.transform(prev_feats.reshape(1, -1))[0]

        window = torch.cat(
            [
                window[:, 1:, :],
                torch.tensor(new_feats_scaled, dtype=torch.float32)
                .reshape(1, 1, -1)
                .to(window.device)
            ],
            dim=1
        )

    prices = [last_close_price * np.exp(np.sum(preds[:i+1])) for i in range(len(preds))]
    return prices


def compute_pct_diff(actual, pred):
    return 100.0 * (pred - actual) / actual


def summarize_metrics(df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    actual = df["actual_price"].values
    pred = df["predicted_price"].values

    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred, squared=False)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)
    dir_acc = np.mean(
        np.sign(np.diff(actual)) == np.sign(np.diff(pred))
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Direction_Acc": dir_acc
    }
