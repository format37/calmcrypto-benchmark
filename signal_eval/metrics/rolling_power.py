"""Rolling Predictive Power Tracker."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def rolling_signal_quality(
    indicator: pd.Series,
    price: pd.Series,
    window: int = 2016  # 7 days at 5min intervals
) -> pd.DataFrame:
    """
    Track how signal quality changes over time.

    This tells you when a signal is "hot" or "cold".

    Args:
        indicator: Signal/indicator series
        price: Price series
        window: Rolling window size (default 2016 = 7 days at 5min)

    Returns:
        DataFrame with:
            - rolling_ic: Rolling Information Coefficient
            - rolling_hit_rate: Rolling directional accuracy
            - ic_stability: Rolling volatility of IC
            - signal_score: Composite score
    """
    returns = price.pct_change()
    fwd_returns = returns.shift(-1)

    # Align series
    aligned = pd.concat([indicator, fwd_returns], axis=1).dropna()
    if len(aligned) < window:
        return pd.DataFrame(
            index=aligned.index,
            columns=['rolling_ic', 'rolling_hit_rate', 'ic_stability', 'signal_score']
        )

    ind = aligned.iloc[:, 0]
    fwd_ret = aligned.iloc[:, 1]

    results = pd.DataFrame(index=aligned.index)

    # Rolling IC
    results['rolling_ic'] = ind.rolling(window).corr(fwd_ret)

    # Rolling hit rate
    signal_dir = np.sign(ind.diff())
    actual_dir = np.sign(fwd_ret)
    correct = (signal_dir == actual_dir).astype(float)
    results['rolling_hit_rate'] = correct.rolling(window).mean()

    # IC stability (lower = more stable)
    results['ic_stability'] = results['rolling_ic'].rolling(window // 2).std()

    # Composite score
    # Higher IC, higher hit rate, lower volatility = better
    ic_component = results['rolling_ic'].abs() * 0.4
    hr_component = results['rolling_hit_rate'] * 0.4
    stability_penalty = results['ic_stability'].fillna(0) * 0.2

    results['signal_score'] = ic_component + hr_component - stability_penalty

    return results


def get_current_power_stats(rolling_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract current (latest) statistics from rolling analysis.

    Args:
        rolling_df: DataFrame from rolling_signal_quality()

    Returns:
        Dict with current values and trends
    """
    if rolling_df.empty or len(rolling_df.dropna()) == 0:
        return {
            'current_ic': np.nan,
            'current_hit_rate': np.nan,
            'current_score': np.nan,
            'ic_trend': np.nan,
            'is_improving': False
        }

    # Get latest non-NaN values
    latest = rolling_df.dropna().iloc[-1]

    # Calculate trend (comparing last 10% to previous)
    n = len(rolling_df.dropna())
    if n > 20:
        recent = rolling_df['signal_score'].dropna().iloc[-n // 10:].mean()
        previous = rolling_df['signal_score'].dropna().iloc[-n // 5:-n // 10].mean()
        ic_trend = recent - previous
        is_improving = ic_trend > 0
    else:
        ic_trend = np.nan
        is_improving = False

    return {
        'current_ic': latest['rolling_ic'],
        'current_hit_rate': latest['rolling_hit_rate'],
        'current_score': latest['signal_score'],
        'ic_trend': ic_trend,
        'is_improving': is_improving
    }
