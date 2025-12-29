"""Generate discrete UP/DOWN trading events from continuous signals.

This module converts continuous signal time series into discrete events
by detecting state transitions (neutral→bullish, bullish→bearish, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

import sys
sys.path.insert(0, '..')

from signal_eval.interpreters import interpret_signal
from signal_eval.config import Config


def generate_signal_events(
    signal: pd.Series,
    signal_name: str,
    config: Config,
    min_confidence: float = 0.4
) -> pd.DataFrame:
    """
    Generate discrete trading events from a continuous signal.

    Uses interpret_signal() from interpreters.py to determine predictions
    at each timestamp, then detects state transitions.

    Args:
        signal: Signal time series with datetime index
        signal_name: Name of the signal (e.g., 'rsi_raw')
        config: Config object with signal_rules
        min_confidence: Minimum confidence to register an event

    Returns:
        DataFrame with columns:
            - timestamp: When the event occurred
            - signal_name: Name of the signal
            - direction: 'UP' or 'DOWN'
            - confidence: 0.0 to 1.0
            - reason: Human-readable explanation
    """
    # Get interpretation rule for this signal
    rule = config.signal_rules.get(signal_name, {'type': 'momentum_directional', 'lookback': 6})

    # Clean signal
    signal = signal.dropna()
    if len(signal) < 10:
        return pd.DataFrame(columns=['timestamp', 'signal_name', 'direction', 'confidence', 'reason'])

    events = []
    prev_state = 0  # Start in neutral state

    # Process each timestamp
    for i in range(len(signal)):
        # Use signal up to current point for interpretation
        signal_window = signal.iloc[:i+1]

        if len(signal_window) < 2:
            continue

        result = interpret_signal(signal_window, rule)
        current_state = result['prediction']
        confidence = result['confidence']
        reason = result['reason']

        # Only register events on state transitions
        if current_state != prev_state and current_state != 0 and confidence >= min_confidence:
            direction = 'UP' if current_state == 1 else 'DOWN'
            events.append({
                'timestamp': signal.index[i],
                'signal_name': signal_name,
                'direction': direction,
                'confidence': confidence,
                'reason': reason
            })

        prev_state = current_state

    df = pd.DataFrame(events)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def generate_signal_events_fast(
    signal: pd.Series,
    signal_name: str,
    config: Config,
    min_confidence: float = 0.4,
    sample_interval: int = 1
) -> pd.DataFrame:
    """
    Faster version of generate_signal_events using vectorized operations where possible.

    For 90 days of 5-min data (~26k points), we sample at intervals to speed up processing.

    Args:
        signal: Signal time series with datetime index
        signal_name: Name of the signal
        config: Config with signal_rules
        min_confidence: Minimum confidence threshold
        sample_interval: Process every Nth point (1 = all, 6 = every 30 min)

    Returns:
        DataFrame with event columns
    """
    rule = config.signal_rules.get(signal_name, {'type': 'momentum_directional', 'lookback': 6})
    rule_type = rule.get('type', 'momentum_directional')

    signal = signal.dropna()
    if len(signal) < 10:
        return pd.DataFrame(columns=['timestamp', 'signal_name', 'direction', 'confidence', 'reason'])

    events = []
    prev_state = 0

    # Different logic based on rule type for efficiency
    if rule_type == 'threshold_contrarian':
        events = _detect_threshold_events(signal, signal_name, rule, min_confidence, sample_interval)
    elif rule_type == 'zscore_contrarian':
        events = _detect_zscore_events(signal, signal_name, rule, min_confidence, sample_interval)
    else:
        # Fall back to point-by-point processing for momentum/level types
        events = _detect_momentum_events(signal, signal_name, rule, min_confidence, sample_interval)

    df = pd.DataFrame(events)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def _detect_threshold_events(
    signal: pd.Series,
    signal_name: str,
    rule: dict,
    min_confidence: float,
    sample_interval: int
) -> List[dict]:
    """Vectorized threshold crossing detection (RSI-style)."""
    bullish_below = rule.get('bullish_below', 30)
    bearish_above = rule.get('bearish_above', 70)

    # Create state series: 1 = oversold (bullish), -1 = overbought (bearish), 0 = neutral
    state = pd.Series(0, index=signal.index)
    state[signal < bullish_below] = 1
    state[signal > bearish_above] = -1

    # Detect transitions
    state_diff = state.diff()
    transition_mask = state_diff != 0
    transition_idx = signal.index[transition_mask]

    events = []
    for idx in transition_idx[::sample_interval]:
        val = signal.loc[idx]
        new_state = state.loc[idx]

        if new_state == 0:
            continue  # Skip transitions to neutral

        # Calculate confidence
        if new_state == 1:
            depth = (bullish_below - val) / bullish_below
            confidence = min(0.9, 0.7 + depth * 0.2)
            reason = f'oversold ({val:.1f})'
        else:
            depth = (val - bearish_above) / (100 - bearish_above)
            confidence = min(0.9, 0.7 + depth * 0.2)
            reason = f'overbought ({val:.1f})'

        if confidence >= min_confidence:
            events.append({
                'timestamp': idx,
                'signal_name': signal_name,
                'direction': 'UP' if new_state == 1 else 'DOWN',
                'confidence': confidence,
                'reason': reason
            })

    return events


def _detect_zscore_events(
    signal: pd.Series,
    signal_name: str,
    rule: dict,
    min_confidence: float,
    sample_interval: int
) -> List[dict]:
    """Vectorized z-score extreme detection."""
    threshold = rule.get('threshold', 2.0)

    # Create state series: 1 = extreme low (bullish), -1 = extreme high (bearish), 0 = normal
    state = pd.Series(0, index=signal.index)
    state[signal < -threshold] = 1
    state[signal > threshold] = -1

    # Detect transitions
    state_diff = state.diff()
    transition_mask = state_diff != 0
    transition_idx = signal.index[transition_mask]

    events = []
    for idx in transition_idx[::sample_interval]:
        val = signal.loc[idx]
        new_state = state.loc[idx]

        if new_state == 0:
            continue

        # Calculate confidence
        extremity = min(abs(val) / threshold, 2.0)
        confidence = min(0.8, 0.5 + (extremity - 1) * 0.3)

        if new_state == 1:
            reason = f'extreme_low (z={val:.2f})'
        else:
            reason = f'extreme_high (z={val:.2f})'

        if confidence >= min_confidence:
            events.append({
                'timestamp': idx,
                'signal_name': signal_name,
                'direction': 'UP' if new_state == 1 else 'DOWN',
                'confidence': confidence,
                'reason': reason
            })

    return events


def _detect_momentum_events(
    signal: pd.Series,
    signal_name: str,
    rule: dict,
    min_confidence: float,
    sample_interval: int
) -> List[dict]:
    """Rolling momentum-based event detection."""
    lookback = rule.get('lookback', 6)
    invert = rule.get('invert', False)

    # Calculate rolling slope
    rolling_change = signal.pct_change(lookback)

    # Determine state based on momentum direction
    state = pd.Series(0, index=signal.index)
    state[rolling_change > 0.001] = 1  # Rising
    state[rolling_change < -0.001] = -1  # Falling

    if invert:
        state = -state

    # Detect transitions
    state_diff = state.diff()
    transition_mask = state_diff != 0
    transition_idx = signal.index[transition_mask]

    events = []
    for idx in transition_idx[::sample_interval]:
        new_state = state.loc[idx]

        if new_state == 0:
            continue

        # Calculate confidence based on momentum magnitude
        change = rolling_change.loc[idx]
        if pd.isna(change):
            continue

        abs_change = abs(change)
        confidence = min(0.7, 0.3 + abs_change * 5)

        reason = 'rising' if new_state == 1 else 'falling'
        if invert:
            reason += ' (inverted)'
        reason += f' (change={change:.4f})'

        if confidence >= min_confidence:
            events.append({
                'timestamp': idx,
                'signal_name': signal_name,
                'direction': 'UP' if new_state == 1 else 'DOWN',
                'confidence': confidence,
                'reason': reason
            })

    return events


def generate_all_signal_events(
    signals: Dict[str, pd.Series],
    config: Config,
    top_n: Optional[int] = None,
    signal_rankings: Optional[pd.DataFrame] = None,
    min_confidence: float = 0.4,
    sample_interval: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Generate events for multiple signals.

    Args:
        signals: Dict mapping signal_name -> Series
        config: Config with signal_rules
        top_n: If set, only process top N signals (requires signal_rankings)
        signal_rankings: DataFrame with signal rankings (must have 'signal_name' column)
        min_confidence: Minimum confidence for events
        sample_interval: Process every Nth point

    Returns:
        Dict mapping signal_name -> events_dataframe
    """
    # Filter to top N signals if specified
    if top_n is not None and signal_rankings is not None and not signal_rankings.empty:
        top_signals = signal_rankings.head(top_n)['signal_name'].tolist()
        signals = {k: v for k, v in signals.items() if k in top_signals}

    all_events = {}
    for signal_name, signal_series in signals.items():
        events = generate_signal_events_fast(
            signal_series,
            signal_name,
            config,
            min_confidence,
            sample_interval
        )
        all_events[signal_name] = events

    return all_events


def merge_all_events(events_by_signal: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge events from all signals into a single DataFrame.

    Args:
        events_by_signal: Dict mapping signal_name -> events_dataframe

    Returns:
        Combined DataFrame sorted by timestamp
    """
    all_dfs = [df for df in events_by_signal.values() if not df.empty]

    if not all_dfs:
        return pd.DataFrame(columns=['timestamp', 'signal_name', 'direction', 'confidence', 'reason'])

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    return merged
