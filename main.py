import pandas as pd
import os, io, time, random, typing as t, requests
from datetime import datetime, timedelta
from binance.client import Client
import time
import numpy as np
from extractors import BinanceFeaturesExtractor, CoinmetricsFeaturesExtractor, BlockchainFeaturesExtractor
import dotenv
dotenv.load_dotenv()
os.environ["PANDAS_USE_NUMEXPR"] = "False"

SYMBOL = "BTCUSDT"
INTERVAL = "1d"
START = "2020-10-01 00:00:00"
END   = "2025-12-31 00:00:00"

have_features_df = False

if not have_features_df:
    binanceExtractor = BinanceFeaturesExtractor(os.environ["api_key_binance"], os.environ["api_secret_binance"])
    blockchainExtractor = BlockchainFeaturesExtractor(START, END)
    coinmetricsExtractor = CoinmetricsFeaturesExtractor(START, END)
    df_binance = binanceExtractor.fetch_futures_klines_binance(SYMBOL, INTERVAL, START, END)
    df_blockchain = blockchainExtractor.run_get_df_blockchain()
    df_valuation = blockchainExtractor.run_get_df_valuation()
    df_coinmetrics, skipped = coinmetricsExtractor.run_get_df()

    date_min = df_blockchain["date"].min().date() if len(df_blockchain) else None
    date_max = df_blockchain["date"].max().date() if len(df_blockchain) else None
    print("\n=== S1 SUMMARY (Blockchain.com — chunked) ===")
    print(f"Rows:  {len(df_blockchain)}  | Range: {date_min} → {date_max}")
    print("Columns:", list(df_blockchain.columns))
    print("\nTop missing:")
    print(df_blockchain.isna().sum().sort_values(ascending=False).head(12))

    date_min = df_valuation["date"].min().date() if len(df_valuation) else None
    date_max = df_valuation["date"].max().date() if len(df_valuation) else None
    print("\n=== S2 SUMMARY (Valuation Proxies) ===")
    print(f"Rows:  {len(df_valuation)}  | Range: {date_min} → {date_max}")
    print("Columns:", list(df_valuation.columns))
    print("\nTop missing:")
    print(df_valuation.isna().sum().sort_values(ascending=False).head(12))


    print("\n=== SUMMARY ===")
    print(f"Rows: {len(df_coinmetrics)}  | Range: {df_coinmetrics['date'].min().date()} → {df_coinmetrics['date'].max().date()}")
    print("Columns:", list(df_coinmetrics.columns))
    if skipped:
        print("Skipped (not available without key / endpoint):", skipped)

    # save all as csv
    df_binance.to_csv("data/df_binance.csv")
    df_blockchain.to_csv("data/df_blockchain.csv")
    df_valuation.to_csv("data/df_valuation.csv")
    df_coinmetrics.to_csv("data/df_coinmetrics.csv")
else:
    df_binance = pd.read_csv("data/df_binance.csv")
    df_blockchain = pd.read_csv("data/df_blockchain.csv")
    df_valuation = pd.read_csv("data/df_valuation.csv")
    df_coinmetrics = pd.read_csv("data/df_coinmetrics.csv")

    if "Unnamed: 0" in df_binance.columns:
        df_binance = df_binance.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in df_blockchain.columns:
        df_blockchain = df_blockchain.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in df_valuation.columns:
        df_valuation = df_valuation.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in df_coinmetrics.columns:
        df_coinmetrics = df_coinmetrics.drop(columns=["Unnamed: 0"])

# ========================= MERGE ALL =========================

def _to_dt(s):
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_localize(None).dt.normalize()

df_binance.columns = [c.strip().lower() for c in df_binance.columns]
df_binance = df_binance.rename(columns={"date": "date"}).sort_values("date").drop_duplicates("date")
df_binance["date"] = _to_dt(df_binance["date"])

if "date" not in df_coinmetrics.columns:
    df_coinmetrics = df_coinmetrics.rename(columns={"time": "date"})
df_coinmetrics["date"] = _to_dt(df_coinmetrics["date"])

df_blockchain["date"] = _to_dt(df_blockchain["date"])

df_valuation["date"] = _to_dt(df_valuation["date"])

# df_economic_features = pd.read_csv("data/economic_features.csv")
# df_economic_features = df_economic_features.rename(columns={"data": "date"})
# df_economic_features["date"] = _to_dt(df_economic_features["date"])
# df_economic_features = df_economic_features.loc[df_economic_features["date"] >= df_binance["date"].min()]

merged = df_binance.merge(df_coinmetrics, on="date", how="outer")
merged = merged.merge(df_blockchain, on="date", how="outer")
merged = merged.merge(df_valuation, on="date", how="outer")
# merged = merged.merge(df_economic_features, on="date", how="outer")

merged = merged.sort_values("date").drop_duplicates("date").reset_index(drop=True)

merged.to_csv("data/merged_btc_features.csv", index=False)
merged.to_pickle("data/merged_btc_features.pkl")

print("=== MERGE SUMMARY ===")
print(f"Rows: {len(merged)}")
print(f"Columns: {len(merged.columns)}")
print(f"Range: {merged['date'].min()} → {merged['date'].max()}")
print("Top columns:", list(merged.columns)[:15])

# =================================================================