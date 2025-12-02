import yfinance as yf
import pandas as pd
import os

ASSETS = {
    "brent": "BZ=F",
    "wti": "CL=F",
    "dxy": "DX-Y.NYB",
    "usdinr": "INR=X",
    "vix": "^VIX",
}

START_DATE = "2007-01-01"
END_DATE = None


def download_all(save_path="../data/raw"):
    os.makedirs(save_path, exist_ok=True)
    df_all = {}
    for name, ticker in ASSETS.items():
        print(f"Downloading {name} ({ticker})...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        df = df.ffill().dropna()
        df.to_csv(os.path.join(save_path, f"{name}.csv"))
        os.makedirs("data", exist_ok=True)
        df.to_csv(f"data/{name}.csv")
        df_all[name] = df
    return df_all


if __name__ == "__main__":
    download_all()
