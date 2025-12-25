#!/usr/bin/env python3
"""
Standalone script for debugging data quality checks.

Usage:
    python -m signal_eval.debug_data_quality
    python -m signal_eval.debug_data_quality --asset BTC --days 14
    python -m signal_eval.debug_data_quality --from-output output/2025-12-25_130630

Set breakpoints in data_quality.py at '# BREAKPOINT:' comments.
"""

import argparse
import pandas as pd
from pathlib import Path

from signal_eval.data_quality import DataQualityChecker, QualityConfig
from signal_eval.data_fetcher import DataFetcher
from signal_eval.signals import SignalRegistry


def load_from_csv(output_dir: str) -> tuple:
    """Load signals from existing output CSVs."""
    path = Path(output_dir)
    signals = {}
    price = None

    for csv_file in path.glob("*_data.csv"):
        signal_name = csv_file.stem.replace("_data", "")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        if "signal_value" in df.columns:
            signals[signal_name] = df["signal_value"]
            if price is None and "btc_price" in df.columns:
                price = df["btc_price"]

    return signals, price


def main():
    parser = argparse.ArgumentParser(description="Debug data quality checks")
    parser.add_argument("--asset", default="SOL", help="Asset symbol (default: SOL)")
    parser.add_argument("--days", type=int, default=7, help="Days of data (default: 7)")
    parser.add_argument("--demo", action="store_true", default=True, help="Use demo data")
    parser.add_argument("--live", action="store_true", help="Use live API data")
    parser.add_argument("--from-output", type=str, help="Load from existing output folder")
    args = parser.parse_args()

    if args.from_output:
        # Load from existing CSVs
        print(f"Loading from: {args.from_output}")
        signals, price = load_from_csv(args.from_output)
        print(f"Loaded {len(signals)} signals")
    else:
        # Fetch fresh data
        use_demo = not args.live
        print(f"Fetching data: asset={args.asset}, days={args.days}, demo={use_demo}")

        fetcher = DataFetcher(demo=use_demo, asset=args.asset)
        raw_data = fetcher.fetch_all(hours=args.days * 24)

        # Build signals
        registry = SignalRegistry.from_raw_data(raw_data)
        signals = registry.all_signals()
        price = registry.get_price_series(raw_data)
        print(f"Built {len(signals)} signals")

    # BREAKPOINT: inspect 'signals' dict and 'price' series before checks
    print(f"\nSignals: {list(signals.keys())}")
    print(f"Price points: {len(price) if price is not None else 0}")

    # Run quality checks
    checker = DataQualityChecker()

    # BREAKPOINT: step into check_all to trace through each check
    report = checker.check_all(signals, price)

    # Print summary
    checker.print_summary(report)

    # BREAKPOINT: inspect 'report' for detailed analysis
    print(f"\n--- Detailed Report ---")
    for name, sig_report in report.signals.items():
        if not sig_report.passed:
            print(f"\n{name}:")
            print(f"  rows={sig_report.total_rows}, nan={sig_report.nan_count}, "
                  f"gaps={sig_report.gap_count}, stale={sig_report.stale_periods}")
            for issue in sig_report.issues[:5]:  # Show first 5 issues
                print(f"  - {issue.issue_type}: {issue.message}")
            if len(sig_report.issues) > 5:
                print(f"  ... and {len(sig_report.issues) - 5} more")

    return report


if __name__ == "__main__":
    main()
