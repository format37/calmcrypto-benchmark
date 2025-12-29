#!/usr/bin/env python
"""Generate interactive signal research reports.

Usage:
    # BTC and ETH, 90 days, 5-min interval, top 5 signals
    python -m signal_research.research_report --assets BTC ETH --days 90 --step 5m --top-signals 5

    # 15-min interval for analysis
    python -m signal_research.research_report --assets BTC --days 90 --step 15m --top-signals 10

    # All assets (takes longer)
    python -m signal_research.research_report --all-assets --days 90

    # Demo mode (no API needed)
    python -m signal_research.research_report --demo --assets BTC --days 7
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard import CalmCryptoAPI
from bulk_data_fetcher import BulkDataFetcher
from signal_eval.data_fetcher import DataFetcher
from signal_eval.signals import SignalRegistry
from signal_eval.evaluator import SignalEvaluator
from signal_eval.config import Config

from signal_research.signal_events import generate_all_signal_events, merge_all_events
from signal_research.chart_builder import ResearchChartBuilder
from signal_research.exporters import (
    export_raw_data_csv,
    export_signal_events_csv,
    export_consolidated_xls,
    create_output_directory,
    export_metadata
)

# Suppress warnings during batch processing
warnings.filterwarnings('ignore')


class SignalResearchReport:
    """Generate interactive signal research reports."""

    def __init__(
        self,
        assets: List[str],
        days: int = 90,
        step: str = '5m',
        top_signals: int = 5,
        output_dir: str = 'output/research',
        demo: bool = False,
        min_confidence: float = 0.4
    ):
        """
        Initialize research report generator.

        Args:
            assets: List of asset symbols to analyze
            days: Days of historical data (default: 90)
            step: Data interval ('5m' or '15m')
            top_signals: Number of top signals to display
            output_dir: Base output directory
            demo: Use demo data instead of API
            min_confidence: Minimum confidence for signal events
        """
        self.assets = assets
        self.days = days
        self.step = step
        self.top_signals = top_signals
        self.output_dir = Path(output_dir)
        self.demo = demo
        self.min_confidence = min_confidence
        self.config = Config()

        self.run_timestamp = datetime.now()
        self.results: Dict[str, Dict[str, Any]] = {}

    def generate_reports(self) -> Dict[str, str]:
        """
        Generate all reports.

        Returns:
            Dict mapping asset -> report path
        """
        # Create output directory
        output_dir = create_output_directory(str(self.output_dir))
        print(f"Output directory: {output_dir}")
        print()

        # Fetch data
        if self.demo:
            all_data = self._fetch_demo_data()
        else:
            all_data = self._fetch_bulk_data()

        if not all_data:
            print("No data fetched. Exiting.")
            return {}

        # Filter to requested assets
        available_assets = [a for a in self.assets if a in all_data]
        missing_assets = [a for a in self.assets if a not in all_data]

        if missing_assets:
            print(f"Warning: Missing data for: {', '.join(missing_assets[:10])}")
            if len(missing_assets) > 10:
                print(f"  ... and {len(missing_assets) - 10} more")
            print()

        print(f"Processing {len(available_assets)} assets...")
        print("-" * 60)

        # Process each asset
        report_paths = {}
        all_assets_export_data = {}

        for i, asset in enumerate(available_assets, 1):
            print(f"[{i}/{len(available_assets)}] {asset}...", end=" ")

            try:
                result = self._process_single_asset(asset, all_data[asset], output_dir)
                if result:
                    report_paths[asset] = result['report_path']
                    all_assets_export_data[asset] = result['export_data']
                    print(f"OK ({result['event_count']} events)")
                else:
                    print("SKIPPED (no events)")
            except Exception as e:
                print(f"FAILED: {e}")

        print("-" * 60)
        print(f"Complete. {len(report_paths)} reports generated.")

        # Export consolidated XLS
        if all_assets_export_data:
            xls_path = output_dir / 'signal_research_report.xlsx'
            try:
                export_consolidated_xls(all_assets_export_data, xls_path)
                print(f"\nConsolidated XLS: {xls_path}")
            except Exception as e:
                print(f"\nFailed to create XLS: {e}")

        # Export metadata
        export_metadata(
            output_dir,
            available_assets,
            self.days,
            self.step,
            self.top_signals,
            self.run_timestamp
        )

        return report_paths

    def _fetch_bulk_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch data using BulkDataFetcher."""
        print(f"Fetching {self.days} days of data at {self.step} intervals...")
        print("Using bulk data fetching (6 API calls for all assets)")
        print()

        fetcher = BulkDataFetcher()
        hours = self.days * 24
        return fetcher.fetch_all_metrics(hours=hours, step=self.step)

    def _fetch_demo_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch demo data for specified assets."""
        print(f"Using demo data for {len(self.assets)} assets...")
        print()

        all_data = {}
        hours = self.days * 24

        for asset in self.assets:
            fetcher = DataFetcher(demo=True, asset=asset)
            raw_data = fetcher.fetch_all(hours=hours, step=self.step)
            if raw_data:
                all_data[asset] = raw_data

        return all_data

    def _process_single_asset(
        self,
        asset: str,
        raw_data: Dict[str, pd.DataFrame],
        output_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Process single asset and generate report.

        Args:
            asset: Asset symbol
            raw_data: Raw metric data
            output_dir: Output directory

        Returns:
            Dict with report_path, event_count, export_data
        """
        # Build signals
        registry = SignalRegistry.from_raw_data(raw_data)
        signals = registry.all_signals()
        price = registry.get_price_series(raw_data)

        if price.empty:
            return None

        # Evaluate signals to get rankings
        try:
            evaluator = SignalEvaluator(price, self.config)
            evaluator.add_signals(signals)
            rankings = evaluator.evaluate_all()
        except Exception:
            rankings = pd.DataFrame()

        # Generate signal events for top N signals
        signal_events = generate_all_signal_events(
            signals,
            self.config,
            top_n=self.top_signals,
            signal_rankings=rankings,
            min_confidence=self.min_confidence
        )

        # Count total events
        total_events = sum(len(df) for df in signal_events.values())
        if total_events == 0:
            # Still generate report with price only
            pass

        # Build chart
        builder = ResearchChartBuilder(asset, self.step)
        fig = builder.build_chart(price, raw_data, signal_events)

        # Save chart
        report_path = output_dir / asset / f'{asset}_research_report.html'
        builder.save_html(str(report_path))

        # Export CSVs
        export_raw_data_csv(raw_data, output_dir, asset)
        export_signal_events_csv(signal_events, output_dir, asset)

        # Prepare export data for XLS
        date_range_start = price.index.min().strftime('%Y-%m-%d') if len(price) > 0 else ''
        date_range_end = price.index.max().strftime('%Y-%m-%d') if len(price) > 0 else ''

        export_data = {
            'raw_data': raw_data,
            'signal_events': signal_events,
            'summary': {
                'asset': asset,
                'date_range_start': date_range_start,
                'date_range_end': date_range_end,
                'total_events': total_events
            }
        }

        return {
            'report_path': str(report_path),
            'event_count': total_events,
            'export_data': export_data
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive signal research reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # BTC and ETH, 90 days, 5-min interval, top 5 signals
    python -m signal_research.research_report --assets BTC ETH --days 90 --step 5m --top-signals 5

    # 15-min interval
    python -m signal_research.research_report --assets BTC --days 90 --step 15m --top-signals 10

    # Demo mode (no API)
    python -m signal_research.research_report --demo --assets BTC --days 7

    # All assets
    python -m signal_research.research_report --all-assets --days 90
        """
    )

    parser.add_argument('--assets', nargs='+', default=['BTC', 'ETH'],
                        help='Assets to analyze (default: BTC ETH)')
    parser.add_argument('--all-assets', action='store_true',
                        help='Process all available assets')
    parser.add_argument('--days', type=int, default=90,
                        help='Days of historical data (default: 90)')
    parser.add_argument('--step', choices=['5m', '15m'], default='5m',
                        help='Data interval (default: 5m)')
    parser.add_argument('--top-signals', type=int, default=5,
                        help='Number of top signals to display (default: 5)')
    parser.add_argument('--min-confidence', type=float, default=0.4,
                        help='Minimum confidence for signal events (default: 0.4)')
    parser.add_argument('--output-dir', default='output/research',
                        help='Output directory (default: output/research)')
    parser.add_argument('--demo', action='store_true',
                        help='Use demo data (no API needed)')

    args = parser.parse_args()

    # Get assets
    if args.all_assets:
        if args.demo:
            print("Cannot use --all-assets with --demo. Using default assets.")
            assets = ['BTC', 'ETH']
        else:
            print("Fetching available assets...")
            api = CalmCryptoAPI()
            assets = api.get_all_assets()
            assets.sort()
            print(f"Found {len(assets)} assets")
    else:
        assets = args.assets

    # Generate reports
    reporter = SignalResearchReport(
        assets=assets,
        days=args.days,
        step=args.step,
        top_signals=args.top_signals,
        output_dir=args.output_dir,
        demo=args.demo,
        min_confidence=args.min_confidence
    )

    reports = reporter.generate_reports()

    # Print summary
    if reports:
        print("\n" + "=" * 60)
        print("GENERATED REPORTS")
        print("=" * 60)
        for asset, path in list(reports.items())[:10]:
            print(f"  {asset}: {path}")
        if len(reports) > 10:
            print(f"  ... and {len(reports) - 10} more")


if __name__ == "__main__":
    main()
