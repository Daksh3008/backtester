import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all OHLCV technical indicators per asset.
    input df: columns MUST contain [Open, High, Low, Close, Volume]
    returns df with appended features
    """
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Stoch %K
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low14) / (high14 - low14)

    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(20).mean()
    md = (tp - ma).abs().rolling(20).mean()
    df['cci_20'] = (tp - ma) / (0.015 * md)

    return df
