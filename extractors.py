import pandas as pd
import os, io, time, random, typing as t, requests
from datetime import datetime, timedelta
from binance.client import Client
import time
import numpy as np
os.environ["PANDAS_USE_NUMEXPR"] = "False"   # avoid optional numexpr import clash

class BinanceFeaturesExtractor:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)

    # ================ Configurações da API Binance ================
    def fetch_futures_klines_binance(self, symbol, interval, start, end):
        """
        Downloads USDT-M futures historical klines (candles)
        and returns a clean DataFrame.
        """
        print(f"Fetching {symbol} futures {interval} candles...")
        klines = self.client.futures_historical_klines(symbol, interval, start_str=start, end_str=end)
        
        if not klines or len(klines) == 0:
            raise ValueError("No data returned. Check symbol, interval, or date range.")
        
        df = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        
        df = df.drop(columns=['Ignore'])
        df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
        df = df.drop(columns=['Open time', 'Close time'])
        
        numeric_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume'
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume', 'Quote asset volume': 'quote_volume',
            'Number of trades': 'trades',
            'Taker buy base asset volume': 'taker_buy_base',
            'Taker buy quote asset volume': 'taker_buy_quote'
        })
        
        df = df[['Date', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
        
        df['symbol'] = symbol
        df['interval'] = interval
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Fetched {len(df)} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")
        return df
    
class BlockchainFeaturesExtractor():
    def __init__(self, START, END):
        self.START = START
        self.END = END
        self.BASE = "https://api.blockchain.info/charts/{name}"
        self.CHARTS_blockchain = {
            "difficulty":        "difficulty",
            "hash-rate":         "hash_rate",
            "blocks-size":       "blk_size_mean",
            "miners-revenue":    "miners_revenue_usd",
            "n-transactions":    "tx_count_day",
        }
        self.CHARTS_valuation = {
            "market-cap":                        "market_cap_usd",
            "estimated-transaction-volume-usd":  "tx_volume_usd",
        }

    def _fetch_chart_once(self, name: str, start: str, end: str, sleep_range=(0.3, 0.7)) -> pd.DataFrame:
        params = {"start": start, "end": end, "format": "csv", "sampled": "false"}
        url = self.BASE.format(name=name)
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"{name} HTTP {r.status_code}: {r.text[:200]}")
        raw = r.text.strip()
        if not raw:
            return pd.DataFrame(columns=["date", "value"])

        df = pd.read_csv(io.StringIO(raw))
        lower = [c.lower() for c in df.columns]
        if "timestamp" in lower:
            ts_col = df.columns[lower.index("timestamp")]
        elif "time" in lower:
            ts_col = df.columns[lower.index("time")]
        elif "date" in lower:
            ts_col = df.columns[lower.index("date")]
        else:
            ts_col = df.columns[0]
            
        val_col = df.columns[lower.index("value")] if "value" in lower else df.columns[-1]

        df = df.rename(columns={ts_col: "date", val_col: "value"})[["date", "value"]]
        dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if dt.isna().all():  # epoch seconds
            df["date"] = pd.to_datetime(df["date"].astype(float), unit="s", utc=True)
        else:
            df["date"] = dt
        df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values("date").drop_duplicates("date")
        m = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        df = df.loc[m].reset_index(drop=True)
        time.sleep(random.uniform(*sleep_range))
        return df

    def _daterange_year_chunks(self, start_str: str, end_str: str):
        start = datetime.fromisoformat(start_str)
        end   = datetime.fromisoformat(end_str)
        cur = start
        while cur <= end:
            # janela anual a partir de 'cur'
            try:
                nxt = datetime(cur.year + 1, cur.month, cur.day) - timedelta(days=1)
            except ValueError:
                # datas como 29/02 → cair para último dia do mês
                last_day = (datetime(cur.year, cur.month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                nxt = datetime(cur.year + 1, last_day.month, last_day.day) - timedelta(days=1)
            chunk_end = min(nxt, end)
            yield cur.date().isoformat(), chunk_end.date().isoformat()
            cur = chunk_end + timedelta(days=1)

    def _fetch_chart_all_years(self, name: str, gstart: str, gend: str) -> pd.DataFrame:
        parts = []
        for s, e in self._daterange_year_chunks(gstart, gend):
            try:
                dfp = self._fetch_chart_once(name, s, e)
                parts.append(dfp)
                print(f"{name:16s}: {s} → {e}  ({len(dfp)} linhas)")
            except Exception as ex:
                print(f"{name:16s}: falha {s}→{e}: {ex}")
        if not parts:
            return pd.DataFrame(columns=["date", "value"])
        df = pd.concat(parts, ignore_index=True)
        return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    def _merge_on_date(self, dfs: t.List[pd.DataFrame]) -> pd.DataFrame:
        out = None
        for d in dfs:
            out = d.copy() if out is None else out.merge(d, on="date", how="outer")
        return out.sort_values("date").reset_index(drop=True)
    
    def run_get_df_blockchain(self):
        frames = []
        for slug, col in self.CHARTS_blockchain.items():
            print(f"Baixando (chunked): {slug} → {col}")
            dfc = self._fetch_chart_all_years(slug, self.START, self.END)
            dfc = dfc.rename(columns={"value": col})
            frames.append(dfc)

        df_blockchain = self._merge_on_date(frames)
        for c in df_blockchain.columns:
            if c != "date":
                df_blockchain[c] = pd.to_numeric(df_blockchain[c], errors="coerce")
        df_blockchain = df_blockchain.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        return df_blockchain
    
    def run_get_df_valuation(self):
        frames = []
        for slug, col in self.CHARTS_valuation.items():
            print(f"Baixando (chunked): {slug:34s} → {col}")
            dfc = self._fetch_chart_all_years(slug, self.START, self.END)
            dfc = dfc.rename(columns={"value": col})
            frames.append(dfc)

        df = self._merge_on_date(frames)
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

        # Deriva NVT = Market Cap / Tx Volume USD
        df["nvt"] = np.where(
            (df["tx_volume_usd"].notna()) & (df["tx_volume_usd"] > 0),
            df["market_cap_usd"] / df["tx_volume_usd"],
            np.nan
        )
        return df

class CoinmetricsFeaturesExtractor:
    def __init__(self, START, END):
        self.START = START
        self.END = END
        self.BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        self.METRICS = [
            ("TxCnt",       "tx_count",          None),
            ("PriceUSD",    "price_usd",         None),
            ("FeeMeanNtv",  "fee_mean_btc",      None),
            ("FeeTotNtv",   "fee_total_btc",     None),
            ("IssContNtv",  "issuance_btc",      None),
            ("SplyCur",     "supply_btc",        None),
            ("HashRate",    "hash_rate",         None),
            ("AdrActCnt",   "active_addresses",  None),
        ]

    def fetch_metric_csv(self, metric: str, start: str, end: str) -> pd.DataFrame:
        """Fetch one metric via CSV; returns df with ['date', metric]. Raises on non-200 or missing 'time'."""
        params = {
            "assets": "btc",
            "metrics": metric,
            "frequency": "1d",
            "start_time": start,
            "end_time": end,
            "format": "csv",
            "page_size": 10000
        }
        r = requests.get(self.BASE, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"{metric} HTTP {r.status_code}: {r.text[:200]}")
        df = pd.read_csv(io.StringIO(r.text))
        if "time" not in df.columns:
            # Some endpoints return 'time' column regardless of order; if missing, show first line to debug.
            raise RuntimeError(f"{metric} unexpected columns: {list(df.columns)[:5]}")
        df = df.rename(columns={"time": "date"})[["date", metric]]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df.sort_values("date").reset_index(drop=True)

    def run_get_df(self):
        merged = None
        ok, skipped = [], []

        for metric, internal, fallback in self.METRICS:
            try_names = [metric] if fallback is None else [metric, fallback]
            got = None
            for name in try_names:
                try:
                    df_m = self.fetch_metric_csv(name, self.START[:10], self.END[:10])
                    df_m = df_m.rename(columns={name: internal})
                    got = df_m[["date", internal]]
                    print(f"✓ {internal:<18} ← {name}")
                    break
                except Exception as e:
                    print(f"… trying {name} failed: {str(e).splitlines()[0]}")
            if got is None:
                print(f"× skipped {internal} (no working name)")
                skipped.append(internal)
                continue

            if merged is None:
                merged = got.copy()
            else:
                merged = merged.merge(got, on="date", how="outer")

        if merged is None or merged.empty:
            raise RuntimeError("No metrics could be fetched. Try again later or with an API key.")

        merged = merged.sort_values("date").reset_index(drop=True)
        return merged, skipped