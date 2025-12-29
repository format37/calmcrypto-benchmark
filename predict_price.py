#!/usr/bin/env python
"""Predict price direction with probability for a given asset."""

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np

from signal_eval.data_fetcher import DataFetcher
from signal_eval.signals import SignalRegistry
from signal_eval.evaluator import SignalEvaluator
from signal_eval.config import Config
from signal_eval.interpreters import (
    interpret_signal,
    compute_timeframe_weight,
    aggregate_predictions,
    compute_dispersion_confidence,
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Timeframes to predict (in 5-min periods)
TIMEFRAMES = {
    '1h': 12,
    '12h': 144,
    '24h': 288,
}


def get_confidence_label(probability: float) -> str:
    """Convert probability to confidence label."""
    p = abs(probability - 0.5) + 0.5  # Distance from 50%
    if p < 0.55:
        return "Very Low"
    elif p < 0.60:
        return "Low"
    elif p < 0.70:
        return "Medium"
    elif p < 0.80:
        return "High"
    else:
        return "Very High"


def predict_asset(asset: str, top_n: int = 5, days: int = 7) -> dict:
    """
    Generate price predictions for an asset.

    Returns dict with predictions for each timeframe.
    """
    # Setup config
    config = Config()
    config.data_hours = days * 24
    config.asset = asset.upper()

    # Fetch data
    fetcher = DataFetcher(demo=False, asset=config.asset)
    try:
        raw_data = fetcher.fetch_all(hours=config.data_hours, step=config.step)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {config.asset}: {e}")

    # Build signals
    registry = SignalRegistry.from_raw_data(raw_data)
    price = registry.get_price_series(raw_data)
    current_price = float(price.iloc[-1])

    # Evaluate signals (suppress output)
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        evaluator = SignalEvaluator(price, config)
        evaluator.add_signals(registry.all_signals())
        rankings = evaluator.evaluate_all()
    finally:
        sys.stdout = old_stdout

    # Get top N signals (optionally filter by Granger significance)
    valid_signals = rankings[rankings['granger_significant'] == True]
    if len(valid_signals) >= top_n:
        top_signals = valid_signals.head(top_n)
    else:
        # Fall back to all signals if not enough Granger-significant ones
        top_signals = rankings.head(top_n)

    # Build predictions for each timeframe
    predictions = {}
    all_signal_details = []

    for timeframe, target_periods in TIMEFRAMES.items():
        timeframe_predictions = []

        for _, row in top_signals.iterrows():
            signal_name = row['signal_name']
            signal = registry.get(signal_name)

            if signal is None or len(signal.dropna()) < 6:
                continue

            # Get signal interpretation rule from config
            rule = config.signal_rules.get(signal_name, {'type': 'momentum_directional'})

            # Interpret signal using rule-based logic
            interp = interpret_signal(signal, rule)

            if interp['prediction'] == 0:
                # Skip neutral signals (no clear prediction)
                continue

            # Get evaluation metrics
            best_lag = row.get('best_lag', 12)
            current_ic = row.get('current_ic', 0.1)
            is_contrarian = row.get('is_contrarian', False)
            effective_hit_rate = row.get('effective_hit_rate', 0.5)

            # Handle NaN in current_ic
            if np.isnan(current_ic):
                current_ic = 0.1

            # Compute timeframe weight (Gaussian based on best_lag proximity)
            timeframe_weight = compute_timeframe_weight(best_lag, target_periods)

            # Skip if signal's optimal lag is too far from this timeframe
            if timeframe_weight < 0.01:
                continue

            # Apply contrarian adjustment from evaluation (in addition to rule-based)
            # This handles cases where historical analysis found opposite relationship
            final_prediction = interp['prediction']
            if is_contrarian:
                final_prediction *= -1

            # Compute signal weight: current_ic (recent strength) × timeframe relevance
            weight = abs(current_ic) * timeframe_weight

            timeframe_predictions.append({
                'signal': signal_name,
                'direction': final_prediction,
                'weight': weight,
                'confidence': interp['confidence'],
                'reason': interp['reason'],
                'timeframe_weight': timeframe_weight,
                'is_contrarian': is_contrarian,
            })

            # Record signal details (only for first timeframe to avoid duplicates)
            if timeframe == '1h':
                all_signal_details.append({
                    'name': signal_name,
                    'interpretation': interp['reason'],
                    'prediction': 'UP' if final_prediction > 0 else 'DOWN',
                    'rule_type': rule.get('type', 'unknown'),
                    'best_lag': best_lag,
                    'current_ic': round(current_ic, 4),
                    'effective_hit_rate': round(effective_hit_rate, 3),
                    'is_contrarian': bool(is_contrarian),
                })

        # Aggregate predictions for this timeframe
        if timeframe_predictions:
            agg = aggregate_predictions(timeframe_predictions)
            disp = compute_dispersion_confidence(timeframe_predictions)

            # Determine final probability (for the predicted direction)
            prob_up = agg['prob_up']
            direction = agg['direction']
            probability = prob_up if direction == 'UP' else (1 - prob_up)

            predictions[timeframe] = {
                'direction': direction,
                'probability': round(probability, 3),
                'confidence': disp['label'],
                'confidence_score': round(disp['confidence'], 3),
                'signals_used': disp['n_signals'],
                'agreement_ratio': round(disp['agreement_ratio'], 3),
            }
        else:
            # No valid predictions for this timeframe
            predictions[timeframe] = {
                'direction': 'NEUTRAL',
                'probability': 0.5,
                'confidence': 'Very Low',
                'confidence_score': 0.0,
                'signals_used': 0,
                'agreement_ratio': 0.5,
            }

    return {
        'asset': config.asset,
        'timestamp': datetime.now().isoformat(),
        'current_price': current_price,
        'predictions': predictions,
        'signals_analyzed': all_signal_details,
    }


def print_prediction(result: dict) -> None:
    """Print prediction results to console."""
    asset = result['asset']
    price = result['current_price']

    print(f"\n{'=' * 60}")
    print(f"  {asset} Price Prediction")
    print(f"{'=' * 60}")
    print(f"Current Price: ${price:,.4f}")
    print(f"Generated: {result['timestamp'][:19]}")
    print()

    # Predictions table with enhanced columns
    print(f"{'Timeframe':<10}{'Direction':<10}{'Prob':<8}{'Confidence':<12}{'Signals':<8}{'Agreement':<10}")
    print("-" * 60)

    for tf, pred in result['predictions'].items():
        direction = pred['direction']
        prob = pred['probability']
        conf = pred['confidence']
        n_signals = pred.get('signals_used', 0)
        agreement = pred.get('agreement_ratio', 0.5)
        print(f"{tf:<10}{direction:<10}{prob:>5.1%}{'':>2}{conf:<12}{n_signals:<8}{agreement:>5.0%}")

    print()
    print("Signal Analysis:")

    # Handle both old format (signals_used) and new format (signals_analyzed)
    signals = result.get('signals_analyzed', result.get('signals_used', []))

    for i, sig in enumerate(signals[:7], 1):
        name = sig['name']
        # New format has 'interpretation', old has 'direction'
        interp = sig.get('interpretation', sig.get('direction', ''))
        pred = sig.get('prediction', '')
        rule_type = sig.get('rule_type', '')
        contr = " [contrarian]" if sig.get('is_contrarian', False) else ""

        print(f"  {i}. {name}")
        print(f"     Rule: {rule_type} | Interpretation: {interp}")
        print(f"     Prediction: {pred}{contr}")

        # Show additional metrics if available
        if 'current_ic' in sig:
            ic = sig['current_ic']
            hr = sig.get('effective_hit_rate', sig.get('hit_rate', 0))
            lag = sig.get('best_lag', 0)
            print(f"     IC: {ic:.4f} | Hit Rate: {hr:.0%} | Best Lag: {lag} periods")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Predict price direction with probability"
    )
    parser.add_argument('asset', help='Asset symbol (e.g., BTC, ETH, SOL)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top signals to use (default: 5)')
    parser.add_argument('--days', type=int, default=14,
                        help='Days of historical data (default: 14)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for JSON file')

    args = parser.parse_args()

    print(f"Fetching data for {args.asset.upper()}...")

    # Generate prediction
    try:
        result = predict_asset(args.asset, top_n=args.top_n, days=args.days)
    except ValueError as e:
        print(f"Error: {e}")
        print("This asset may not have complete data available.")
        if args.days < 14:
            print(f"Try with more days: python predict_price.py {args.asset} --days 14")
        return

    # Print to console
    print_prediction(result)

    # Save JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / f"prediction_{result['asset']}.json"

    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
