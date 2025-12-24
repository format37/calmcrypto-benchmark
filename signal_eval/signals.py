"""Signal definitions for evaluation."""

from typing import Dict, Callable
import pandas as pd
import numpy as np


class SignalRegistry:
    """Registry for defining and computing trading signals from raw data."""

    def __init__(self):
        self._signals: Dict[str, pd.Series] = {}
        self._definitions: Dict[str, Callable] = {}

    def register(self, name: str, signal: pd.Series) -> None:
        """Register a computed signal."""
        self._signals[name] = signal

    def get(self, name: str) -> pd.Series:
        """Get a registered signal."""
        return self._signals.get(name)

    def all_signals(self) -> Dict[str, pd.Series]:
        """Get all registered signals."""
        return self._signals.copy()

    def names(self) -> list:
        """Get list of all signal names."""
        return list(self._signals.keys())

    @classmethod
    def from_raw_data(cls, data: Dict[str, pd.DataFrame]) -> 'SignalRegistry':
        """
        Build all signals from raw data.

        Args:
            data: Dict with price, total_borrow, total_repay, rsi,
                  funding_rate, open_interest DataFrames

        Returns:
            SignalRegistry with all computed signals
        """
        registry = cls()

        # Extract series from DataFrames
        price = data['price']['price']
        borrow = data['total_borrow']['total_borrow']
        repay = data['total_repay']['total_repay']
        rsi = data['rsi']['rsi']
        funding = data['funding_rate']['funding_rate']
        oi = data['open_interest']['open_interest']

        # 1. Borrow/Repay Ratio
        registry.register('borrow_repay_ratio', borrow / repay)

        # 2. Borrow Momentum (1hr = 12 periods at 5min)
        registry.register('borrow_momentum', borrow.pct_change(12))

        # 3. Repay Momentum
        registry.register('repay_momentum', repay.pct_change(12))

        # 4. Raw RSI
        registry.register('rsi_raw', rsi)

        # 5. RSI Z-Score (standardized RSI)
        rsi_mean = rsi.rolling(288).mean()  # 1-day rolling mean
        rsi_std = rsi.rolling(288).std()
        rsi_zscore = (rsi - rsi_mean) / rsi_std
        registry.register('rsi_zscore', rsi_zscore)

        # 6. Total Borrow (raw volume)
        registry.register('total_borrow', borrow)

        # 7. Total Repay (raw volume)
        registry.register('total_repay', repay)

        # 8. Funding Rate
        registry.register('funding_rate', funding)

        # 9. Open Interest
        registry.register('open_interest', oi)

        # 10. Open Interest Momentum
        registry.register('oi_momentum', oi.pct_change(12))

        # Additional derived signals

        # 11. Net Flow (Borrow - Repay)
        registry.register('net_flow', borrow - repay)

        # 12. Net Flow Momentum
        net_flow = borrow - repay
        registry.register('net_flow_momentum', net_flow.pct_change(12))

        # 13. Borrow Repay Ratio Momentum
        ratio = borrow / repay
        registry.register('ratio_momentum', ratio.pct_change(12))

        # 14. Funding Rate Z-Score
        funding_mean = funding.rolling(288).mean()
        funding_std = funding.rolling(288).std()
        funding_zscore = (funding - funding_mean) / funding_std
        registry.register('funding_zscore', funding_zscore)

        # 15. OI Z-Score
        oi_mean = oi.rolling(288).mean()
        oi_std = oi.rolling(288).std()
        oi_zscore = (oi - oi_mean) / oi_std
        registry.register('oi_zscore', oi_zscore)

        return registry

    def get_price_series(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Extract price series from raw data."""
        return data['price']['price']


def align_series(*series: pd.Series) -> tuple:
    """Align multiple series to common index, dropping NaN."""
    df = pd.concat(series, axis=1).dropna()
    return tuple(df.iloc[:, i] for i in range(len(series)))
