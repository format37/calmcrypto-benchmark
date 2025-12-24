"""Lead-Lag Cross-Correlation analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def lead_lag_analysis(
    indicator: pd.Series,
    price: pd.Series,
    max_lag: int = 48
) -> Dict[str, Any]:
    """
    Calculate cross-correlation at different lags to find which indicator leads price.

    Positive lag = indicator leads price
    Negative lag = price leads indicator

    Args:
        indicator: Signal/indicator series
        price: Price series
        max_lag: Maximum lag to test (in periods, default 48 = 4 hours at 5min)

    Returns:
        Dict with:
            - best_lag: Lag with highest absolute correlation
            - best_correlation: Correlation at best lag
            - lead_lag_score: Normalized score (0-1) based on how well indicator leads
            - correlation_df: DataFrame with lag and correlation columns
    """
    returns = price.pct_change()

    # Align series
    aligned = pd.concat([indicator, returns], axis=1).dropna()
    if len(aligned) < max_lag * 2:
        return {
            'best_lag': 0,
            'best_correlation': np.nan,
            'lead_lag_score': 0.0,
            'correlation_df': pd.DataFrame()
        }

    ind = aligned.iloc[:, 0]
    ret = aligned.iloc[:, 1]

    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # Indicator leads: shift indicator back, compare with current returns
            corr = ind.shift(lag).corr(ret)
        elif lag < 0:
            # Price leads: compare indicator with past returns
            corr = ind.corr(ret.shift(-lag))
        else:
            corr = ind.corr(ret)

        results.append({'lag': lag, 'correlation': corr})

    df = pd.DataFrame(results)

    # Find best predictive lag (positive lag = indicator leads)
    best_idx = df['correlation'].abs().idxmax()
    best_lag = df.loc[best_idx, 'lag']
    best_correlation = df.loc[best_idx, 'correlation']

    # Calculate lead-lag score:
    # Higher score if best correlation is at positive lag (indicator leads)
    # and correlation is strong
    if best_lag > 0 and not np.isnan(best_correlation):
        # Indicator leads price - this is what we want
        lead_score = abs(best_correlation) * (1 + best_lag / max_lag) / 2
    elif best_lag == 0:
        # Concurrent - moderate value
        lead_score = abs(best_correlation) * 0.3 if not np.isnan(best_correlation) else 0
    else:
        # Price leads indicator - not useful for prediction
        lead_score = abs(best_correlation) * 0.1 if not np.isnan(best_correlation) else 0

    return {
        'best_lag': int(best_lag),
        'best_correlation': best_correlation,
        'lead_lag_score': min(lead_score, 1.0),  # Cap at 1.0
        'correlation_df': df
    }
