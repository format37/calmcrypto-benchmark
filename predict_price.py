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

    # Get top N signals
    top_signals = rankings.head(top_n)

    # Get current signal directions and build predictions
    signals_used = []
    predictions = {}

    for timeframe, periods in TIMEFRAMES.items():
        up_confidence = 0.0
        down_confidence = 0.0

        for _, row in top_signals.iterrows():
            signal_name = row['signal_name']
            signal = registry.get(signal_name)

            if signal is None or len(signal) < 2:
                continue

            # Current signal direction (last value vs previous)
            current_val = signal.iloc[-1]
            prev_val = signal.iloc[-2]

            if np.isnan(current_val) or np.isnan(prev_val):
                continue

            signal_rising = current_val > prev_val
            signal_direction = "rising" if signal_rising else "falling"

            # Check if contrarian
            is_contrarian = row.get('is_contrarian', False)
            hit_rate = row.get('effective_hit_rate', 0.5)
            composite = row.get('composite_score', 0)

            # Determine predicted direction
            if is_contrarian:
                predicts_up = not signal_rising
            else:
                predicts_up = signal_rising

            predicted_dir = "UP" if predicts_up else "DOWN"

            # Weight by composite score and hit rate
            weight = composite * hit_rate

            if predicts_up:
                up_confidence += weight
            else:
                down_confidence += weight

            # Record signal info (only once, for first timeframe)
            if timeframe == '1h':
                signals_used.append({
                    'name': signal_name,
                    'direction': signal_direction,
                    'prediction': predicted_dir,
                    'hit_rate': round(hit_rate, 3),
                    'is_contrarian': bool(is_contrarian),
                })

        # Calculate probability
        total = up_confidence + down_confidence
        if total > 0:
            prob_up = up_confidence / total
        else:
            prob_up = 0.5

        direction = "UP" if prob_up > 0.5 else "DOWN"
        probability = prob_up if direction == "UP" else (1 - prob_up)

        predictions[timeframe] = {
            'direction': direction,
            'probability': round(probability, 3),
            'confidence': get_confidence_label(probability),
        }

    return {
        'asset': config.asset,
        'timestamp': datetime.now().isoformat(),
        'current_price': current_price,
        'predictions': predictions,
        'signals_used': signals_used,
    }


def print_prediction(result: dict) -> None:
    """Print prediction results to console."""
    asset = result['asset']
    price = result['current_price']

    print(f"\n{'=' * 50}")
    print(f"  {asset} Price Prediction")
    print(f"{'=' * 50}")
    print(f"Current Price: ${price:,.4f}")
    print(f"Generated: {result['timestamp'][:19]}")
    print()

    # Predictions table
    print(f"{'Timeframe':<12}{'Direction':<12}{'Probability':<14}{'Confidence':<12}")
    print("-" * 50)

    for tf, pred in result['predictions'].items():
        direction = pred['direction']
        prob = pred['probability']
        conf = pred['confidence']
        print(f"{tf:<12}{direction:<12}{prob:>6.1%}{'':>7}{conf:<12}")

    print()
    print("Top Contributing Signals:")
    for i, sig in enumerate(result['signals_used'][:5], 1):
        name = sig['name']
        direction = sig['direction']
        pred = sig['prediction']
        hr = sig['hit_rate']
        contr = " (contrarian)" if sig['is_contrarian'] else ""
        print(f"  {i}. {name} ({direction}) -> {pred} ({hr:.0%} hit rate{contr})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Predict price direction with probability"
    )
    parser.add_argument('asset', help='Asset symbol (e.g., BTC, ETH, SOL)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top signals to use (default: 5)')
    parser.add_argument('--days', type=int, default=7,
                        help='Days of historical data (default: 7)')
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
