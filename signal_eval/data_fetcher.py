"""Data fetching for signal evaluation."""

import sys
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to import from dashboard
sys.path.insert(0, str(Path(__file__).parent.parent))
from dashboard import CalmCryptoAPI


class DataFetcher:
    """Fetch raw data from CalmCrypto API or generate demo data."""

    def __init__(self, demo: bool = False, asset: str = "BTC"):
        """
        Initialize data fetcher.

        Args:
            demo: Use demo data instead of live API
            asset: Asset symbol to analyze (e.g., "BTC", "ETH", "SOL")
        """
        self.demo = demo
        self.asset = asset.upper()
        # Futures symbol (e.g., BTC -> BTCUSDT)
        self.futures_symbol = f"{self.asset}USDT"
        if not demo:
            self.api = CalmCryptoAPI()

    def fetch_all(self, hours: int = 168, step: str = "5m") -> Dict[str, pd.DataFrame]:
        """
        Fetch all raw data needed for signal evaluation.

        Returns:
            Dict with keys: price, total_borrow, total_repay, rsi,
                           funding_rate, open_interest
        """
        if self.demo:
            return self._generate_demo_data(hours, step)

        return self._fetch_live_data(hours, step)

    def _fetch_live_data(self, hours: int, step: str) -> Dict[str, pd.DataFrame]:
        """Fetch live data from API for specified asset."""
        data = {}

        # Asset Price
        data['price'] = self.api.get_price(self.asset, hours)
        data['price'].columns = ['price']

        # Asset-specific Borrow
        borrow_result = self.api.query_range(
            f'binance_24h_total_borrow_usdt{{asset="{self.asset}"}}', hours, step
        )
        data['total_borrow'] = self.api.to_dataframe(borrow_result)
        data['total_borrow'].columns = ['total_borrow']

        # Asset-specific Repay
        repay_result = self.api.query_range(
            f'binance_24h_total_repay_usdt{{asset="{self.asset}"}}', hours, step
        )
        data['total_repay'] = self.api.to_dataframe(repay_result)
        data['total_repay'].columns = ['total_repay']

        # RSI
        data['rsi'] = self.api.get_rsi(self.asset, "3m", hours)
        data['rsi'].columns = ['rsi']

        # Funding Rate (futures)
        data['funding_rate'] = self.api.get_funding(self.futures_symbol, hours)
        data['funding_rate'].columns = ['funding_rate']

        # Open Interest (futures)
        data['open_interest'] = self.api.get_oi(self.futures_symbol, hours)
        data['open_interest'].columns = ['open_interest']

        # Store asset info in metadata
        data['_asset'] = self.asset

        return data

    def _generate_demo_data(self, hours: int, step: str) -> Dict[str, pd.DataFrame]:
        """Generate realistic demo data for testing."""
        # Parse step to minutes
        if step.endswith('m'):
            step_minutes = int(step[:-1])
        elif step.endswith('h'):
            step_minutes = int(step[:-1]) * 60
        else:
            step_minutes = 5

        n_points = (hours * 60) // step_minutes
        end_time = datetime.now()
        timestamps = pd.date_range(
            end=end_time, periods=n_points, freq=f'{step_minutes}min'
        )

        np.random.seed(42)  # Reproducible demo data

        # BTC Price: random walk with trend
        base_price = 90000
        returns = np.random.normal(0.0001, 0.002, n_points)
        trend = np.linspace(0, 0.05, n_points)  # 5% upward trend
        price_series = base_price * np.exp(np.cumsum(returns) + trend)

        # Add some correlation between signals and price for realistic demo
        price_changes = np.diff(price_series, prepend=price_series[0])

        # Total Borrow: correlated with price volatility
        borrow_base = 2.5e9
        borrow_noise = np.random.normal(0, 0.02, n_points)
        borrow_signal = np.abs(price_changes) / base_price * 10  # Volatility driven
        total_borrow = borrow_base * (1 + borrow_noise + borrow_signal)

        # Total Repay: slightly lagged and anti-correlated
        repay_base = 2.3e9
        repay_noise = np.random.normal(0, 0.02, n_points)
        repay_signal = np.roll(np.abs(price_changes) / base_price * 8, 3)
        total_repay = repay_base * (1 + repay_noise + repay_signal)

        # RSI: mean-reverting around 50, with some leading signal
        rsi_base = 50
        rsi_noise = np.random.normal(0, 5, n_points)
        # RSI tends to rise before price rises
        rsi_lead = np.roll(returns, -6) * 1000
        rsi = np.clip(rsi_base + rsi_noise + rsi_lead, 10, 90)

        # Funding Rate: small values, correlated with price momentum
        funding_base = 0.0001
        funding_noise = np.random.normal(0, 0.00005, n_points)
        momentum = pd.Series(price_series).pct_change(12).fillna(0).values
        funding_rate = funding_base + funding_noise + momentum * 0.001

        # Open Interest: grows with price, drops with volatility
        oi_base = 15e9
        oi_trend = np.linspace(0, 0.1, n_points)
        oi_noise = np.random.normal(0, 0.01, n_points)
        oi_volatility = -np.abs(price_changes) / base_price * 5
        open_interest = oi_base * (1 + oi_trend + oi_noise + oi_volatility)

        # Create DataFrames
        data = {
            'price': pd.DataFrame(
                {'price': price_series}, index=timestamps
            ),
            'total_borrow': pd.DataFrame(
                {'total_borrow': total_borrow}, index=timestamps
            ),
            'total_repay': pd.DataFrame(
                {'total_repay': total_repay}, index=timestamps
            ),
            'rsi': pd.DataFrame(
                {'rsi': rsi}, index=timestamps
            ),
            'funding_rate': pd.DataFrame(
                {'funding_rate': funding_rate}, index=timestamps
            ),
            'open_interest': pd.DataFrame(
                {'open_interest': open_interest}, index=timestamps
            ),
            '_asset': self.asset,  # Store asset info
        }

        return data
