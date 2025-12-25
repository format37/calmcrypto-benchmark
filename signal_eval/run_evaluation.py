#!/usr/bin/env python3
"""
Signal Evaluation CLI

Evaluates trading signals against BTC price movements and outputs
ranked results to CSV files.

Usage:
    python -m signal_eval.run_evaluation --demo
    python -m signal_eval.run_evaluation --days 7 --top-n 10
"""

import argparse
import sys
from pathlib import Path

from .config import Config
from .data_fetcher import DataFetcher
from .signals import SignalRegistry
from .evaluator import SignalEvaluator
from .output import save_evaluation_results
from .report import generate_report
from .loader import load_from_output
from .data_quality import DataQualityChecker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trading signals for predictive power"
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Use demo data instead of live API'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of data to fetch (default: 7)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Number of top signals to output (default: from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for CSV files (default: output)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to config file (default: config.json)'
    )
    parser.add_argument(
        '--include-rolling',
        action='store_true',
        help='Include rolling signal quality CSVs'
    )
    parser.add_argument(
        '--save-config',
        action='store_true',
        help='Save default config to config.json'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate interactive HTML benchmark report'
    )
    parser.add_argument(
        '--from-output',
        type=str,
        default=None,
        metavar='DIR',
        help='Generate report from existing output directory (skip data fetching)'
    )
    parser.add_argument(
        '--asset',
        type=str,
        default='BTC',
        help='Asset symbol to analyze (e.g., BTC, ETH, SOL). Default: BTC'
    )

    return parser.parse_args()


def print_summary(rankings, output_summary):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("SIGNAL EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTop {len(rankings)} Signals by Composite Score:")
    print("-" * 72)

    # Print table header
    print(f"{'Rank':<6}{'Signal':<22}{'Score':>8}{'IC':>8}{'EffHR':>8}{'Type':>10}")
    print("-" * 72)

    for idx, row in rankings.iterrows():
        name = row['signal_name'][:21]
        score = row['composite_score']
        ic = row.get('best_spearman_ic', 0)
        hr = row.get('hit_rate', 0.5)
        is_contr = row.get('is_contrarian', False)

        if not isinstance(ic, (int, float)):
            ic = 0
        if not isinstance(hr, (int, float)):
            hr = 0.5

        # Calculate effective hit rate
        eff_hr = max(hr, 1 - hr)
        signal_type = "CONTR" if is_contr else "direct"

        print(f"{idx:<6}{name:<22}{score:>8.3f}{ic:>8.3f}{eff_hr:>8.1%}{signal_type:>10}")

    print("-" * 72)

    # Output files
    print(f"\nOutput Directory: {output_summary['run_directory']}")
    print(f"Files Saved: {output_summary['files_saved']}")
    print("\nCSV Files:")
    for f in output_summary['file_list'][:10]:  # Show first 10
        print(f"  - {Path(f).name}")
    if len(output_summary['file_list']) > 10:
        print(f"  ... and {len(output_summary['file_list']) - 10} more")

    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    # Load or create config
    config = Config.load(args.config)

    if args.save_config:
        config.save(args.config)
        print(f"Config saved to {args.config}")
        return

    # Override config with CLI args
    if args.top_n:
        config.top_n = args.top_n
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.days:
        config.data_hours = args.days * 24
    if args.asset:
        config.asset = args.asset.upper()

    # Handle --from-output mode (generate report from existing data)
    if args.from_output:
        print("Signal Evaluation System")
        print("-" * 40)
        print(f"Mode: Load from existing output")
        print(f"Source: {args.from_output}")
        print("-" * 40)

        print("\nLoading data from CSV files...")
        try:
            evaluator = load_from_output(args.from_output)
            rankings = evaluator.evaluate_all()
            print(f"  Loaded {len(rankings)} signals")

            # Print quick summary
            print("\nSignal Rankings:")
            for idx, row in rankings.head(config.top_n).iterrows():
                name = row['signal_name'][:24]
                score = row.get('composite_score', 0)
                print(f"  {idx}. {name:<25} {score:.3f}")

            # Generate report (no asset info when loading from existing data)
            print("\nGenerating interactive report...")
            report_path = generate_report(
                evaluator,
                output_dir=config.output_dir,
                top_n=config.top_n,
                asset=None  # Will use random 4-digit ID
            )
            print(f"  Report saved to: {report_path}")
            return [report_path]

        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Normal mode: fetch and evaluate
    print("Signal Evaluation System")
    print("-" * 40)
    print(f"Mode: {'Demo' if args.demo else 'Live API'}")
    print(f"Asset: {config.asset}")
    print(f"Data: {args.days} days")
    print(f"Top N: {config.top_n}")
    print("-" * 40)

    # Fetch data
    print("\nFetching data...")
    fetcher = DataFetcher(demo=args.demo, asset=config.asset)
    raw_data = fetcher.fetch_all(hours=config.data_hours, step=config.step)

    print(f"  {config.asset} Price: {len(raw_data['price'])} points")
    print(f"  Range: ${raw_data['price']['price'].min():,.0f} - "
          f"${raw_data['price']['price'].max():,.0f}")

    # Build signals
    print("\nBuilding signals...")
    registry = SignalRegistry.from_raw_data(raw_data)
    print(f"  Signals registered: {len(registry.names())}")
    for name in registry.names():
        print(f"    - {name}")

    # Evaluate
    print("\nEvaluating signals...")
    price = registry.get_price_series(raw_data)
    evaluator = SignalEvaluator(price, config)
    evaluator.add_signals(registry.all_signals())

    rankings = evaluator.evaluate_all()

    # Data quality checks
    print("\nRunning data quality checks...")
    quality_checker = DataQualityChecker()
    quality_report = quality_checker.check_all(registry.all_signals(), price)
    quality_checker.print_summary(quality_report)

    # Save results
    print("\nSaving results...")
    output_summary = save_evaluation_results(
        evaluator,
        output_dir=config.output_dir,
        top_n=config.top_n,
        include_rolling=args.include_rolling,
        asset=config.asset,
        data_hours=config.data_hours,
        step=config.step,
        quality_report=quality_report
    )

    # Print summary
    print_summary(rankings.head(config.top_n), output_summary)

    # Generate interactive report if requested
    if args.report:
        print("\nGenerating interactive report...")
        report_path = generate_report(
            evaluator,
            output_dir=config.output_dir,
            top_n=config.top_n,
            asset=config.asset
        )
        print(f"  Report saved to: {report_path}")

    # Return file list for programmatic use
    return output_summary['file_list']


if __name__ == '__main__':
    main()
