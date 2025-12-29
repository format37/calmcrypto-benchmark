"""Signal interpretation functions for price prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any


def interpret_signal(signal: pd.Series, rule: dict) -> Dict[str, Any]:
    """
    Apply interpretation rule to signal's current value.

    Args:
        signal: Signal time series
        rule: Interpretation rule dict with 'type' and type-specific params

    Returns:
        dict with:
            - prediction: +1 (bullish), -1 (bearish), or 0 (neutral)
            - confidence: 0.0 to 1.0 (how confident in this prediction)
            - reason: Human-readable explanation
    """
    rule_type = rule.get('type', 'momentum_directional')

    # Get clean signal data
    clean = signal.dropna()
    if len(clean) < 2:
        return {'prediction': 0, 'confidence': 0, 'reason': 'insufficient_data'}

    current_val = clean.iloc[-1]

    if rule_type == 'threshold_contrarian':
        return _interpret_threshold_contrarian(current_val, rule)

    elif rule_type == 'zscore_contrarian':
        return _interpret_zscore_contrarian(current_val, rule)

    elif rule_type == 'momentum_directional':
        return _interpret_momentum_directional(clean, rule)

    elif rule_type == 'level':
        return _interpret_level(current_val, rule)

    elif rule_type == 'level_with_extremes':
        return _interpret_level_with_extremes(current_val, rule)

    else:
        # Unknown type - fall back to momentum
        return _interpret_momentum_directional(clean, {'lookback': 6})


def _interpret_threshold_contrarian(current_val: float, rule: dict) -> Dict[str, Any]:
    """
    Threshold-based contrarian interpretation (RSI-style).
    Values below bullish_threshold predict price up.
    Values above bearish_threshold predict price down.
    """
    bullish_below = rule.get('bullish_below', 30)
    bearish_above = rule.get('bearish_above', 70)
    neutral_range = rule.get('neutral_range', [35, 65])

    if np.isnan(current_val):
        return {'prediction': 0, 'confidence': 0, 'reason': 'nan_value'}

    if current_val < bullish_below:
        # Deeply oversold - strong bullish
        depth = (bullish_below - current_val) / bullish_below
        confidence = min(0.9, 0.7 + depth * 0.2)
        return {'prediction': 1, 'confidence': confidence, 'reason': f'oversold ({current_val:.1f})'}

    elif current_val > bearish_above:
        # Deeply overbought - strong bearish
        depth = (current_val - bearish_above) / (100 - bearish_above)
        confidence = min(0.9, 0.7 + depth * 0.2)
        return {'prediction': -1, 'confidence': confidence, 'reason': f'overbought ({current_val:.1f})'}

    elif neutral_range[0] <= current_val <= neutral_range[1]:
        # Clearly in neutral zone
        return {'prediction': 0, 'confidence': 0.2, 'reason': f'neutral ({current_val:.1f})'}

    else:
        # Between thresholds but not fully neutral - weak signal
        if current_val < neutral_range[0]:
            return {'prediction': 1, 'confidence': 0.4, 'reason': f'slightly_oversold ({current_val:.1f})'}
        else:
            return {'prediction': -1, 'confidence': 0.4, 'reason': f'slightly_overbought ({current_val:.1f})'}


def _interpret_zscore_contrarian(current_val: float, rule: dict) -> Dict[str, Any]:
    """
    Z-score based contrarian interpretation.
    Extreme negative z-scores predict price up (reversal).
    Extreme positive z-scores predict price down (reversal).
    """
    threshold = rule.get('threshold', 2.0)

    if np.isnan(current_val):
        return {'prediction': 0, 'confidence': 0, 'reason': 'nan_value'}

    if current_val < -threshold:
        # Extreme low - expect bounce
        extremity = min(abs(current_val) / threshold, 2.0)  # Cap at 2x threshold
        confidence = min(0.8, 0.5 + (extremity - 1) * 0.3)
        return {'prediction': 1, 'confidence': confidence, 'reason': f'extreme_low (z={current_val:.2f})'}

    elif current_val > threshold:
        # Extreme high - expect pullback
        extremity = min(current_val / threshold, 2.0)
        confidence = min(0.8, 0.5 + (extremity - 1) * 0.3)
        return {'prediction': -1, 'confidence': confidence, 'reason': f'extreme_high (z={current_val:.2f})'}

    else:
        # Normal range - weak/no signal
        confidence = 0.3 * (1 - abs(current_val) / threshold)  # Lower confidence near edges
        return {'prediction': 0, 'confidence': confidence, 'reason': f'normal_range (z={current_val:.2f})'}


def _interpret_momentum_directional(signal: pd.Series, rule: dict) -> Dict[str, Any]:
    """
    Momentum-based directional interpretation.
    Rising signal = bullish, falling signal = bearish.
    Uses slope over lookback window.
    """
    lookback = rule.get('lookback', 6)
    invert = rule.get('invert', False)

    # Get recent window
    window_size = min(lookback, len(signal))
    if window_size < 2:
        return {'prediction': 0, 'confidence': 0, 'reason': 'insufficient_data'}

    recent = signal.iloc[-window_size:]

    # Calculate slope using linear regression
    x = np.arange(len(recent))
    y = recent.values

    # Handle NaN in window
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return {'prediction': 0, 'confidence': 0, 'reason': 'too_many_nans'}

    x_clean, y_clean = x[mask], y[mask]

    # Simple slope calculation
    if len(x_clean) >= 2:
        slope = (y_clean[-1] - y_clean[0]) / (len(y_clean) - 1)
        # Normalize by mean absolute value to get relative change
        mean_val = np.mean(np.abs(y_clean))
        if mean_val > 0:
            relative_slope = slope / mean_val
        else:
            relative_slope = 0
    else:
        return {'prediction': 0, 'confidence': 0, 'reason': 'insufficient_clean_data'}

    # Determine direction
    direction = 1 if relative_slope > 0 else -1
    if invert:
        direction *= -1

    # Confidence based on slope magnitude
    abs_slope = abs(relative_slope)
    confidence = min(0.7, 0.3 + abs_slope * 10)  # Scale relative slope to confidence

    reason = 'rising' if (relative_slope > 0) != invert else 'falling'
    return {'prediction': direction, 'confidence': confidence, 'reason': f'{reason} (slope={relative_slope:.4f})'}


def _interpret_level(current_val: float, rule: dict) -> Dict[str, Any]:
    """
    Simple level-based interpretation.
    Above threshold = bullish, below threshold = bearish.
    """
    bullish_above = rule.get('bullish_above', 0)
    bearish_below = rule.get('bearish_below', 0)

    if np.isnan(current_val):
        return {'prediction': 0, 'confidence': 0, 'reason': 'nan_value'}

    if current_val > bullish_above:
        # Above bullish threshold
        excess = current_val - bullish_above
        confidence = min(0.6, 0.4 + abs(excess) * 0.1)
        return {'prediction': 1, 'confidence': confidence, 'reason': f'above_threshold ({current_val:.4f})'}

    elif current_val < bearish_below:
        # Below bearish threshold
        deficit = bearish_below - current_val
        confidence = min(0.6, 0.4 + abs(deficit) * 0.1)
        return {'prediction': -1, 'confidence': confidence, 'reason': f'below_threshold ({current_val:.4f})'}

    else:
        # Between thresholds - neutral
        return {'prediction': 0, 'confidence': 0.3, 'reason': f'between_thresholds ({current_val:.4f})'}


def _interpret_level_with_extremes(current_val: float, rule: dict) -> Dict[str, Any]:
    """
    Level-based interpretation with contrarian behavior at extremes.
    Normal range: above 0 = bullish, below 0 = bearish.
    Extreme values: contrarian (extreme high = bearish, extreme low = bullish).
    """
    bullish_above = rule.get('bullish_above', 0)
    bearish_below = rule.get('bearish_below', 0)
    extreme_threshold = rule.get('extreme_threshold', 0.001)
    extreme_contrarian = rule.get('extreme_contrarian', True)

    if np.isnan(current_val):
        return {'prediction': 0, 'confidence': 0, 'reason': 'nan_value'}

    # Check for extreme values first
    if extreme_contrarian:
        if current_val > extreme_threshold:
            # Extreme high - contrarian bearish
            return {'prediction': -1, 'confidence': 0.7, 'reason': f'extreme_high_contrarian ({current_val:.6f})'}
        elif current_val < -extreme_threshold:
            # Extreme low - contrarian bullish
            return {'prediction': 1, 'confidence': 0.7, 'reason': f'extreme_low_contrarian ({current_val:.6f})'}

    # Normal level interpretation
    if current_val > bullish_above:
        return {'prediction': 1, 'confidence': 0.5, 'reason': f'positive ({current_val:.6f})'}
    elif current_val < bearish_below:
        return {'prediction': -1, 'confidence': 0.5, 'reason': f'negative ({current_val:.6f})'}
    else:
        return {'prediction': 0, 'confidence': 0.2, 'reason': f'neutral ({current_val:.6f})'}


def compute_timeframe_weight(best_lag: int, target_periods: int, sigma_factor: float = 0.5) -> float:
    """
    Gaussian weighting based on how well signal's best_lag matches the target timeframe.

    Args:
        best_lag: Signal's optimal prediction horizon (in 5-min periods)
        target_periods: Timeframe we're predicting (e.g., 12 for 1h, 144 for 12h)
        sigma_factor: Width of Gaussian (0.5 = sigma is half of target)

    Returns:
        Weight between 0 and 1. High when best_lag is close to target_periods.
    """
    if target_periods <= 0:
        return 0.0

    sigma = max(target_periods * sigma_factor, 1)  # Minimum sigma of 1
    distance = abs(best_lag - target_periods)
    weight = np.exp(-0.5 * (distance / sigma) ** 2)

    return float(weight)


def aggregate_predictions(predictions: list) -> Dict[str, Any]:
    """
    Aggregate individual signal predictions into final probability.

    Args:
        predictions: List of dicts with 'direction' (+1/-1), 'weight', 'confidence'

    Returns:
        dict with prob_up (0-1), direction ('UP'/'DOWN'/'NEUTRAL'), raw_score (-1 to +1)
    """
    if not predictions:
        return {'prob_up': 0.5, 'direction': 'NEUTRAL', 'raw_score': 0}

    # Compute weighted sum
    weighted_sum = 0.0
    total_weight = 0.0

    for pred in predictions:
        # Effective weight = signal weight × interpretation confidence
        eff_weight = pred.get('weight', 0.5) * pred.get('confidence', 0.5)
        weighted_sum += pred['direction'] * eff_weight
        total_weight += eff_weight

    if total_weight == 0:
        return {'prob_up': 0.5, 'direction': 'NEUTRAL', 'raw_score': 0}

    # Normalized score: -1 to +1
    raw_score = weighted_sum / total_weight

    # Convert to probability using sigmoid
    # Scale factor controls how quickly we approach 0/1
    scale = 3.0  # raw_score of 0.5 -> prob ~0.82
    prob_up = 1 / (1 + np.exp(-scale * raw_score))

    direction = 'UP' if prob_up > 0.5 else ('DOWN' if prob_up < 0.5 else 'NEUTRAL')

    return {
        'prob_up': float(prob_up),
        'direction': direction,
        'raw_score': float(raw_score)
    }


def compute_dispersion_confidence(predictions: list) -> Dict[str, Any]:
    """
    Compute confidence based on agreement/disagreement among signals.

    High confidence = signals agree (low dispersion)
    Low confidence = signals disagree (high dispersion)

    Args:
        predictions: List of prediction dicts with 'direction' (+1/-1) and 'weight'

    Returns:
        dict with confidence (0-1), agreement_ratio (0.5-1), label (str)
    """
    if len(predictions) < 2:
        return {
            'confidence': 0.3,
            'agreement_ratio': 1.0,
            'label': 'Low',
            'n_signals': len(predictions)
        }

    # Count weighted votes
    up_weight = sum(p.get('weight', 0.5) for p in predictions if p['direction'] == 1)
    down_weight = sum(p.get('weight', 0.5) for p in predictions if p['direction'] == -1)
    total_weight = up_weight + down_weight

    if total_weight == 0:
        return {
            'confidence': 0.1,
            'agreement_ratio': 0.5,
            'label': 'Very Low',
            'n_signals': len(predictions)
        }

    # Agreement ratio: how lopsided is the vote? (0.5 = split, 1.0 = unanimous)
    majority_weight = max(up_weight, down_weight)
    agreement_ratio = majority_weight / total_weight

    # Sample size factor: more signals = more confidence
    n_signals = len(predictions)
    sample_factor = min(1.0, n_signals / 5)  # Full confidence at 5+ signals

    # Combined confidence
    # Agreement contributes 70%, sample size 30%
    confidence = (agreement_ratio - 0.5) * 2 * 0.7 + sample_factor * 0.3
    confidence = max(0, min(1, confidence))

    # Label
    if confidence < 0.2:
        label = 'Very Low'
    elif confidence < 0.4:
        label = 'Low'
    elif confidence < 0.6:
        label = 'Medium'
    elif confidence < 0.8:
        label = 'High'
    else:
        label = 'Very High'

    return {
        'confidence': float(confidence),
        'agreement_ratio': float(agreement_ratio),
        'label': label,
        'n_signals': n_signals
    }
