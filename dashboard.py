import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class CalmCryptoAPI:
    def __init__(self):
        self.base_url = os.getenv("GRAFANA_URL", "https://grafana.calmcrypto.app")
        self.ds_uid = os.getenv("GRAFANA_DS_UID", "victoriametrics-uid")
        self.auth = (
            os.getenv("GRAFANA_USER"),
            os.getenv("GRAFANA_PASSWORD")
        )
    
    def query(self, expr: str) -> dict:
        """Instant query - returns latest value"""
        url = f"{self.base_url}/api/datasources/proxy/uid/{self.ds_uid}/api/v1/query"
        return requests.get(url, params={"query": expr}, auth=self.auth).json()
    
    def query_range(self, expr: str, hours: int = 24, step: str = "5m") -> dict:
        """Range query - returns time series"""
        url = f"{self.base_url}/api/datasources/proxy/uid/{self.ds_uid}/api/v1/query_range"
        end = datetime.now()
        start = end - timedelta(hours=hours)
        params = {
            "query": expr,
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "step": step
        }
        return requests.get(url, params=params, auth=self.auth).json()
    
    def to_dataframe(self, result: dict) -> pd.DataFrame:
        """Convert range query result to DataFrame"""
        if result["status"] != "success":
            raise ValueError(f"Query failed: {result}")
        
        data = result["data"]["result"]
        if not data:
            return pd.DataFrame()
        
        # Handle matrix (range) results
        if result["data"]["resultType"] == "matrix":
            series = data[0]
            df = pd.DataFrame(series["values"], columns=["timestamp", "value"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["value"] = df["value"].astype(float)
            df.set_index("timestamp", inplace=True)
            return df
        
        # Handle vector (instant) results
        return pd.DataFrame([
            {"metric": r["metric"], "value": float(r["value"][1])}
            for r in data
        ])
    
    # Convenience methods
    def get_price(self, asset: str, hours: int = 24) -> pd.DataFrame:
        return self.to_dataframe(
            self.query_range(f'binance_price_usdt{{asset="{asset}"}}', hours)
        )
    
    def get_hands(self, asset: str, hours: int = 24) -> pd.DataFrame:
        expr = f'binance_24h_total_borrow_usdt{{asset="{asset}"}}-binance_24h_total_repay_usdt{{asset="{asset}"}}'
        return self.to_dataframe(self.query_range(expr, hours))
    
    def get_oi(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        return self.to_dataframe(
            self.query_range(f'binance_futures_open_interest{{symbol="{symbol}"}}', hours)
        )
    
    def get_funding(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        return self.to_dataframe(
            self.query_range(f'binance_futures_funding_rate{{symbol="{symbol}"}}', hours)
        )
    
    def get_rsi(self, symbol: str, timeframe: str = "3m", hours: int = 24) -> pd.DataFrame:
        expr = f'rsi{{symbol="{symbol}", timeframe="{timeframe}", source="indicator_core"}}'
        return self.to_dataframe(self.query_range(expr, hours))
    
    def get_all_assets(self) -> list:
        """Get list of all available assets"""
        result = self.query('binance_price_usdt')
        return [r["metric"]["asset"] for r in result["data"]["result"]]


# Usage example
if __name__ == "__main__":
    api = CalmCryptoAPI()
    
    # Get BTC data
    btc_price = api.get_price("BTC", hours=168)  # 7 days
    btc_hands = api.get_hands("BTC", hours=168)
    btc_oi = api.get_oi("BTCUSDT", hours=168)
    
    print(f"BTC Price: ${btc_price['value'].iloc[-1]:,.2f}")
    print(f"BTC Hands: ${btc_hands['value'].iloc[-1]:,.0f}")
    print(f"Available assets: {api.get_all_assets()[:10]}...")
