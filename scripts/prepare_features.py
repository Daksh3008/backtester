# scripts/prepare_features.py

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.technicals import compute_indicators

ASSETS = {
    "brent": "BZ=F",
    "wti": "CL=F",
    "dxy": "DX-Y.NYB",
    "usdinr": "INR=X",
    "vix": "^VIX",
}

# ---------------------------------------------------------------
# DATE-PARSING SAFE CSV LOADER
# ---------------------------------------------------------------
def load_asset_file(path: str, asset_name: str, ticker: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Missing file: {path}")

    df = pd.read_csv(
        path,
        header=[0, 1],      # field + ticker
        index_col=0,
        dayfirst=True,
    )

    df.index.name = "date"

    # normalize to string
    df.index = df.index.astype(str).str.strip()
    # force datetime
    df.index = pd.to_datetime(df.index, errors="coerce")

    before = len(df)
    df = df[~df.index.isna()]
    if len(df) < before:
        print(f"[WARN] Dropped {before - len(df)} rows with bad dates from {asset_name}")

    df = df.sort_index()

    wanted = ["Open","High","Low","Close","Volume","Price"]
    simple = pd.DataFrame(index=df.index)

    for field in wanted:
        col = (field, ticker)
        if col in df.columns:
            simple[field] = df[col]

    required = ["Open","High","Low","Close","Volume"]
    missing = [x for x in required if x not in simple.columns]
    if missing:
        raise ValueError(f"{asset_name} missing OHLCV: {missing}")

    return simple


# ---------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------

def rolling_vol(ret: pd.Series, windows=[5,10,20]):
    out = {}
    for w in windows:
        out[f"vol_{w}"] = ret.rolling(w).std()
    return pd.DataFrame(out)

def rolling_ma_ret(ret: pd.Series, windows=[5,10]):
    out = {}
    for w in windows:
        out[f"ma_ret_{w}"] = ret.rolling(w).mean()
    return pd.DataFrame(out)

def return_slope(ret: pd.Series, windows=[5,10]):
    """
    Linear regression slope over last w returns.
    If any value in window is NaN or inf -> slope = NaN.
    """
    out = {}
    lr = LinearRegression()
    arr = ret.values
    idx = ret.index

    for w in windows:
        slopes = []
        for i in range(len(arr)):
            if i < w:
                slopes.append(np.nan)
                continue

            window_vals = arr[i-w:i]

            # Hard guards against any invalid numbers
            if (
                np.isnan(window_vals).any()
                or np.isinf(window_vals).any()
                or len(window_vals) != w
            ):
                slopes.append(np.nan)
                continue

            X = np.arange(w).reshape(-1,1)
            y = window_vals.reshape(-1,1)

            # final guard: y must be finite
            if not np.isfinite(y).all():
                slopes.append(np.nan)
                continue

            lr.fit(X, y)
            slopes.append(lr.coef_[0][0])

        out[f"ret_slope_{w}"] = slopes

    return pd.DataFrame(out, index=idx)


def atr(df: pd.DataFrame, period=14):
    # Expect columns: Open, High, Low, Close
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

def garman_klass_vol(df: pd.DataFrame):
    # expects Open, High, Low, Close
    return 0.5 * np.log(df["High"]/df["Low"])**2 - (2*np.log(2)-1)*(np.log(df["Close"]/df["Open"]))**2

def obv(df: pd.DataFrame):
    close = df["Close"]
    vol = df["Volume"]
    direction = np.sign(close.diff())
    return (direction * vol).fillna(0).cumsum()

def avg_vol_ratio(df: pd.DataFrame, period=20):
    return df["Volume"] / df["Volume"].rolling(period).mean()

def pct_distance_MA(df: pd.DataFrame, periods=[20,50,100]):
    out = {}
    close = df["Close"]
    for p in periods:
        ma = close.rolling(p).mean()
        out[f"pct_dist_ma{p}"] = (close - ma) / ma
    return pd.DataFrame(out)

def zscore(df: pd.Series, window=50):
    rmean = df.rolling(window).mean()
    rstd = df.rolling(window).std()
    return (df - rmean) / rstd

def rolling_entropy(ret: pd.Series, window=20):
    """
    Shannon entropy over return distribution in rolling window.
    Skip windows with NaN/Inf -> return NaN.
    """
    arr = ret.values
    out = []

    for i in range(len(arr)):
        if i < window:
            out.append(np.nan)
            continue

        w = arr[i-window:i]

        # skip if invalid
        if np.isnan(w).any() or np.isinf(w).any():
            out.append(np.nan)
            continue

        # histogram
        hist, _ = np.histogram(w, bins=10, density=True)

        # skip degenerate hist
        hist = hist[hist > 0]
        if len(hist) == 0:
            out.append(np.nan)
            continue

        entropy = -np.sum(hist * np.log(hist))
        out.append(entropy)

    return pd.Series(out, index=ret.index)


def ema_slope(series: pd.Series, period):
    ema = series.ewm(span=period, adjust=False).mean()
    lr = LinearRegression()
    slopes = []
    arr = ema.values
    for i in range(len(arr)):
        if i < period:
            slopes.append(np.nan)
            continue
        y = arr[i-period:i].reshape(-1,1)
        X = np.arange(period).reshape(-1,1)
        lr.fit(X,y)
        slopes.append(lr.coef_[0][0])
    return pd.Series(slopes, index=series.index)


# ---------------------------------------------------------------
# Main Builder
# ---------------------------------------------------------------

def build_feature_matrix(raw_path="data") -> pd.DataFrame:
    asset_frames = {}

    # Load + base features
    for asset_name, ticker in ASSETS.items():
        file = os.path.join(raw_path, f"{asset_name}.csv")
        base = load_asset_file(file, asset_name, ticker)

        # compute base returns
        base["ret"] = np.log(base["Close"] / base["Close"].shift(1))

        # TA from your previous compute_indicators
        ta = compute_indicators(base[["Open","High","Low","Close","Volume"]].copy())

        # Rolling volatility
        vol_df = rolling_vol(base["ret"])

        # Rolling MA of returns
        ma_ret = rolling_ma_ret(base["ret"])

        # Return momentum slope
        slope = return_slope(base["ret"])

        # ATR
        atr_df = atr(base)

        # Garman Klass
        gk = garman_klass_vol(base)

        # OBV
        obv_df = obv(base)

        # Avg volume ratio
        avgvr = avg_vol_ratio(base)

        # pct dist from MA
        pct_ma = pct_distance_MA(base)

        # zscore on Close
        zsc = zscore(base["Close"])

        # rolling entropy of returns
        entr = rolling_entropy(base["ret"])

        # ema slopes
        ema100 = ema_slope(base["Close"], 100)
        ema200 = ema_slope(base["Close"], 200)

        df = pd.concat([
            base,
            ta[["rsi_14","macd","macd_signal","stoch_k","cci_20"]],
            vol_df,
            ma_ret,
            slope,
            atr_df.rename("atr_14"),
            gk.rename("gk_vol"),
            obv_df.rename("obv"),
            avgvr.rename("volume_ratio"),
            pct_ma,
            zsc.rename("zscore_50"),
            entr.rename("ret_entropy_20"),
            ema100.rename("ema100_slope"),
            ema200.rename("ema200_slope")
        ], axis=1)

        asset_frames[asset_name] = df.add_prefix(f"{asset_name}_")

    # merge all
    full = pd.concat(asset_frames.values(), axis=1).sort_index()
    full = full.ffill()

    # Brent close target
    full["close"] = full["brent_Close"].astype(float)
    full["log_ret"] = np.log(full["close"] / full["close"].shift(1))


    # ---------------------------------------------------------
    # Cross-asset correlations vs Brent
    # ---------------------------------------------------------
    for asset_name in ASSETS:
        if asset_name == "brent": 
            continue
        full[f"corr20_brent_{asset_name}"] = (
            full["brent_ret"].rolling(20).corr(full[f"{asset_name}_ret"])
        )
        full[f"corr60_brent_{asset_name}"] = (
            full["brent_ret"].rolling(60).corr(full[f"{asset_name}_ret"])
        )

    # ---------------------------------------------------------
    # Spread ratios
    # ---------------------------------------------------------
    for asset_name in ASSETS:
        if asset_name == "brent":
            continue
        full[f"spread_ratio_{asset_name}"] = full["brent_Close"] / full[f"{asset_name}_Close"]

    # Return spread
    for asset_name in ASSETS:
        if asset_name == "brent":
            continue
        full[f"ret_spread_{asset_name}"] = full["brent_ret"] - full[f"{asset_name}_ret"]

    # Macro pressures
    full["oil_fx_pressure"] = full["brent_ret"] - full["usdinr_ret"]
    full["oil_usd_pressure"] = full["brent_ret"] - full["dxy_ret"]
    full["risk_off_pressure"] = full["brent_ret"] - full["vix_ret"]

    # ---------------------------------------------------------
    # Calendar features
    # ---------------------------------------------------------
    dt = full.index
    full["dow"] = dt.weekday
    full["dom"] = dt.day
    full["month"] = dt.month
    full["is_month_end"] = dt.is_month_end.astype(int)
    full["is_month_start"] = dt.is_month_start.astype(int)

    # ---------------------------------------------------------
    # FINAL STEP: HANDLE NaNs (your rule)
    # ---------------------------------------------------------

    na_mask = full.isna()

    # Backfill from future data
    full = full.bfill()

    # scale only filled NA values
    full = full.mask(na_mask, full * 0.001)

    # in case any remain (rare)
    full = full.fillna(1e-9)

    return full


def save_feature_matrix(df, path="data/feature_matrix.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"[INFO] Saved feature matrix to {path} | shape={df.shape}")


if __name__ == "__main__":
    print("[INFO] Building enriched feature matrix...")
    fm = build_feature_matrix()
    save_feature_matrix(fm)
