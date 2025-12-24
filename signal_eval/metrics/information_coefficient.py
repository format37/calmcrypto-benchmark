"""Information Coefficient (IC) calculation."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    periods: int = 1,
    rolling_window: int = 288
) -> Dict[str, Any]:
    """
    Calculate Information Coefficient: correlation between signal and future returns.

    Args:
        signal: Indicator values
        forward_returns: Price returns N periods ahead
        periods: How far ahead to look (default 1 period)
        rolling_window: Window for rolling IC calculation (default 288 = 1 day at 5min)

    Returns:
        Dict with:
            - pearson_ic: Pearson correlation coefficient
            - spearman_ic: Spearman rank correlation (captures non-linear)
            - ic_ir: Information Ratio (mean / std of rolling IC)
            - ic_mean: Mean of rolling IC
            - ic_std: Std of rolling IC
            - rolling_ic: Series of rolling IC values
    """
    # Shift returns forward (future returns at time t are returns from t to t+periods)
    fwd_ret = forward_returns.shift(-periods)

    # Align series
    aligned = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(aligned) < rolling_window:
        return {
            'pearson_ic': np.nan,
            'spearman_ic': np.nan,
            'ic_ir': np.nan,
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'rolling_ic': pd.Series(dtype=float)
        }

    sig = aligned.iloc[:, 0]
    ret = aligned.iloc[:, 1]

    # Pearson IC (linear relationship)
    pearson_ic = sig.corr(ret)

    # Spearman IC (rank correlation - captures non-linear)
    spearman_ic = sig.corr(ret, method='spearman')

    # Rolling IC for stability analysis
    rolling_ic = sig.rolling(window=rolling_window).corr(ret)

    # IC Information Ratio (mean / std) — higher = more consistent signal
    ic_mean = rolling_ic.mean()
    ic_std = rolling_ic.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0

    return {
        'pearson_ic': pearson_ic,
        'spearman_ic': spearman_ic,
        'ic_ir': ic_ir,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'rolling_ic': rolling_ic
    }
