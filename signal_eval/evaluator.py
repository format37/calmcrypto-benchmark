"""Signal Evaluator - main evaluation engine."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from .config import Config
from .metrics import (
    calculate_ic,
    lead_lag_analysis,
    calculate_hit_rate,
    granger_test,
    rolling_signal_quality,
)
from .metrics.rolling_power import get_current_power_stats


class SignalEvaluator:
    """Evaluate and rank trading signals based on predictive power."""

    def __init__(self, price: pd.Series, config: Optional[Config] = None):
        """
        Initialize evaluator with reference price series.

        Args:
            price: BTC price series (or other reference asset)
            config: Configuration object (uses defaults if None)
        """
        self.price = price
        self.returns = price.pct_change()
        self.config = config or Config()
        self.signals: Dict[str, pd.Series] = {}
        self.results: Dict[str, Dict] = {}

    def add_signal(self, name: str, signal: pd.Series) -> None:
        """Register a signal for evaluation."""
        self.signals[name] = signal

    def add_signals(self, signals: Dict[str, pd.Series]) -> None:
        """Register multiple signals."""
        self.signals.update(signals)

    def evaluate_signal(
        self,
        name: str,
        forward_periods: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single signal across all metrics.

        Args:
            name: Signal name
            forward_periods: List of forward periods to test

        Returns:
            Dict with all metrics for this signal
        """
        if name not in self.signals:
            raise ValueError(f"Signal '{name}' not registered")

        signal = self.signals[name]
        fwd_periods = forward_periods or self.config.forward_periods
        period_labels = self.config.forward_period_labels()

        result = {
            'signal_name': name,
            'metrics': {},
            'by_period': {}
        }

        # Per-period metrics
        for period in fwd_periods:
            label = period_labels.get(period, f"{period}p")

            # Information Coefficient
            ic_result = calculate_ic(
                signal, self.returns, period, self.config.rolling_window
            )

            result['by_period'][label] = {
                'period': period,
                'pearson_ic': ic_result['pearson_ic'],
                'spearman_ic': ic_result['spearman_ic'],
                'ic_ir': ic_result['ic_ir'],
            }

        # Lead-Lag Analysis (once, not per period)
        lead_lag_result = lead_lag_analysis(
            signal, self.price, self.config.max_lag
        )

        # Hit Rate
        hit_rate_result = calculate_hit_rate(signal, self.returns)

        # Granger Causality
        granger_result = granger_test(
            signal, self.price, self.config.granger_max_lag
        )

        # Rolling Power (use 1-day window to have enough data points for rolling calc)
        rolling_result = rolling_signal_quality(
            signal, self.price, self.config.rolling_window
        )
        power_stats = get_current_power_stats(rolling_result)

        # Aggregate metrics
        result['metrics'] = {
            # Best IC across periods
            'best_pearson_ic': max(
                (r['pearson_ic'] for r in result['by_period'].values()),
                key=lambda x: abs(x) if not np.isnan(x) else 0
            ),
            'best_spearman_ic': max(
                (r['spearman_ic'] for r in result['by_period'].values()),
                key=lambda x: abs(x) if not np.isnan(x) else 0
            ),
            'best_ic_ir': max(
                (r['ic_ir'] for r in result['by_period'].values()),
                key=lambda x: x if not np.isnan(x) else -999
            ),

            # Lead-Lag
            'best_lag': lead_lag_result['best_lag'],
            'lead_lag_correlation': lead_lag_result['best_correlation'],
            'lead_lag_score': lead_lag_result['lead_lag_score'],

            # Hit Rate
            'hit_rate': hit_rate_result['overall_hit_rate'],
            'hit_rate_bullish': hit_rate_result['hit_rate_bullish'],
            'hit_rate_bearish': hit_rate_result['hit_rate_bearish'],
            'total_signals': hit_rate_result['total_signals'],
            # Contrarian detection: if hit_rate < 0.5, signal predicts opposite
            'is_contrarian': hit_rate_result['overall_hit_rate'] < 0.5 if not np.isnan(hit_rate_result['overall_hit_rate']) else False,
            'effective_hit_rate': max(hit_rate_result['overall_hit_rate'], 1 - hit_rate_result['overall_hit_rate']) if not np.isnan(hit_rate_result['overall_hit_rate']) else 0.5,

            # Granger
            'granger_best_lag': granger_result['best_lag'],
            'granger_p_value': granger_result['best_p_value'],
            'granger_significant': granger_result['significant'],
            'granger_score': granger_result['granger_score'],

            # Rolling Power
            'current_ic': power_stats['current_ic'],
            'current_hit_rate': power_stats['current_hit_rate'],
            'current_score': power_stats['current_score'],
            'is_improving': power_stats['is_improving'],
        }

        # Calculate composite score
        result['composite_score'] = self._calculate_composite_score(result['metrics'])

        # Store rolling data for later
        result['rolling_data'] = rolling_result

        return result

    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted composite score from metrics."""
        weights = self.config.weights

        # Normalize each component to 0-1 range
        components = {}

        # IC component (use absolute spearman IC, capped at 0.3 -> 1.0)
        spearman = metrics.get('best_spearman_ic', 0)
        if np.isnan(spearman):
            spearman = 0
        components['ic'] = min(abs(spearman) / 0.3, 1.0)

        # IC-IR component (capped at 2.0 -> 1.0)
        ic_ir = metrics.get('best_ic_ir', 0)
        if np.isnan(ic_ir):
            ic_ir = 0
        components['ic_ir'] = min(max(ic_ir, 0) / 2.0, 1.0)

        # Hit rate - use effective hit rate (handles contrarian signals)
        # If hit_rate < 0.5, signal is contrarian (invert for 1 - hit_rate accuracy)
        hit_rate = metrics.get('hit_rate', 0.5)
        if np.isnan(hit_rate):
            hit_rate = 0.5
        # Effective hit rate: max(hit_rate, 1 - hit_rate)
        # This captures value of both directional AND contrarian signals
        effective_hit_rate = max(hit_rate, 1 - hit_rate)
        # Transform: 0.5 -> 0, 1.0 -> 1
        components['hit_rate'] = (effective_hit_rate - 0.5) * 2

        # Lead-lag score (already 0-1)
        lead_lag = metrics.get('lead_lag_score', 0)
        if np.isnan(lead_lag):
            lead_lag = 0
        components['lead_lag'] = lead_lag

        # Granger score (already 0-1)
        granger = metrics.get('granger_score', 0)
        if np.isnan(granger):
            granger = 0
        components['granger'] = granger

        # Calculate weighted sum
        score = sum(
            weights.get(key, 0) * value
            for key, value in components.items()
        )

        return score

    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all registered signals.

        Returns:
            DataFrame with all signals ranked by composite score
        """
        results = []

        for name in self.signals:
            try:
                result = self.evaluate_signal(name)
                self.results[name] = result

                # Flatten for DataFrame
                row = {
                    'signal_name': name,
                    'composite_score': result['composite_score'],
                    **result['metrics']
                }
                results.append(row)
            except Exception as e:
                print(f"Error evaluating signal '{name}': {e}")
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('composite_score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
            df = df.set_index('rank')

        return df

    def get_top_n(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        Get top N signals by composite score.

        Args:
            n: Number of signals to return (uses config.top_n if None)

        Returns:
            DataFrame with top N signals
        """
        if not self.results:
            self.evaluate_all()

        n = n or self.config.top_n
        df = self.evaluate_all()
        return df.head(n)

    def get_signal_data(self, name: str) -> pd.DataFrame:
        """
        Get signal data with forward returns for CSV export.

        Args:
            name: Signal name

        Returns:
            DataFrame with timestamp, signal value, price, and forward returns
        """
        if name not in self.signals:
            raise ValueError(f"Signal '{name}' not registered")

        signal = self.signals[name]
        fwd_periods = self.config.forward_periods
        period_labels = self.config.forward_period_labels()

        # Build DataFrame
        df = pd.DataFrame({
            'signal_value': signal,
            'btc_price': self.price
        })

        # Add forward returns
        for period in fwd_periods:
            label = period_labels.get(period, f"{period}p")
            df[f'forward_return_{label}'] = self.returns.shift(-period)

        return df.dropna()
