"""Hit Rate and Directional Accuracy calculation."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_hit_rate(
    signal: pd.Series,
    returns: pd.Series,
    threshold: float = 0
) -> Dict[str, Any]:
    """
    Calculate what percentage of time the signal correctly predicts price direction.

    Args:
        signal: Indicator values
        returns: Price returns series
        threshold: Minimum return magnitude to consider (default 0)

    Returns:
        Dict with:
            - overall_hit_rate: Overall directional accuracy
            - hit_rate_bullish: Hit rate when signal predicts up
            - hit_rate_bearish: Hit rate when signal predicts down
            - total_signals: Number of signals evaluated
    """
    # Signal direction: change in signal
    signal_direction = np.sign(signal.diff())

    # Actual direction: next period return
    actual_direction = np.sign(returns.shift(-1))

    # Align series
    aligned = pd.concat([signal_direction, actual_direction], axis=1).dropna()
    if len(aligned) < 10:
        return {
            'overall_hit_rate': np.nan,
            'hit_rate_bullish': np.nan,
            'hit_rate_bearish': np.nan,
            'total_signals': 0
        }

    sig_dir = aligned.iloc[:, 0]
    act_dir = aligned.iloc[:, 1]

    # Filter by threshold if specified
    if threshold > 0:
        mask = returns.shift(-1).abs() > threshold
        aligned_mask = mask.reindex(aligned.index).fillna(False)
        sig_dir = sig_dir[aligned_mask]
        act_dir = act_dir[aligned_mask]

    # Hit rate: correct predictions
    correct = (sig_dir == act_dir)
    hit_rate = correct.mean() if len(correct) > 0 else np.nan

    # Conditional hit rates
    up_signals = sig_dir > 0
    down_signals = sig_dir < 0

    hit_rate_up = np.nan
    hit_rate_down = np.nan

    if up_signals.sum() > 0:
        hit_rate_up = (correct & up_signals).sum() / up_signals.sum()

    if down_signals.sum() > 0:
        hit_rate_down = (correct & down_signals).sum() / down_signals.sum()

    return {
        'overall_hit_rate': hit_rate,
        'hit_rate_bullish': hit_rate_up,
        'hit_rate_bearish': hit_rate_down,
        'total_signals': len(sig_dir)
    }
