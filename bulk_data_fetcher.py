#!/usr/bin/env python
"""
Bulk data fetcher that retrieves all CalmCrypto data in minimal API calls.

Instead of making 2400+ sequential API calls (6 calls per asset x 400 assets),
this fetches all data in just 6 bulk queries (one per metric type).

Usage:
    # Fetch 7 days of data, export to CSV
    python bulk_data_fetcher.py --hours 168 --step 5m --export-csv

    # Fetch 24 hours, print summary only
    python bulk_data_fetcher.py --hours 24
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from dashboard import CalmCryptoAPI

load_dotenv()


class BulkDataFetcher:
    """Fetch all CalmCrypto data in 6 API calls instead of 2400+."""

    # Bulk queries - one per metric type, returns ALL assets
    BULK_QUERIES = {
        'price': {
            'query': 'binance_price_usdt',
            'label_key': 'asset',
            'normalize': 'spot'  # BTC -> BTC
        },
        'total_borrow': {
            'query': 'binance_24h_total_borrow_usdt',
            'label_key': 'asset',
            'normalize': 'spot'
        },
        'total_repay': {
            'query': 'binance_24h_total_repay_usdt',
            'label_key': 'asset',
            'normalize': 'spot'
        },
        'rsi': {
            'query': 'rsi{timeframe="3m", source="indicator_core"}',
            'label_key': 'symbol',
            'normalize': 'spot'  # BTC -> BTC
        },
        'open_interest': {
            'query': 'binance_futures_open_interest',
            'label_key': 'symbol',
            'normalize': 'futures'  # BTCUSDT -> BTC
        },
        'funding_rate': {
            'query': 'binance_futures_funding_rate',
            'label_key': 'symbol',
            'normalize': 'futures'  # BTCUSDT -> BTC
        },
    }

    def __init__(self):
        self.api = CalmCryptoAPI()
        self._raw_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.errors: List[Dict[str, Any]] = []
        self.fetch_time: Optional[datetime] = None
        self.hours: int = 0
        self.step: str = ""

    def fetch_all_metrics(self, hours: int = 168, step: str = "5m") -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch all metrics for all assets in 6 API calls.

        Args:
            hours: Hours of historical data (default: 168 = 7 days)
            step: Data interval (default: '5m')

        Returns:
            Dict mapping asset -> metric -> DataFrame
            Compatible with DataFetcher.fetch_all() format
        """
        self.fetch_time = datetime.now()
        self.hours = hours
        self.step = step
        self.errors = []
        self._raw_data = {}

        print(f"Bulk fetching {hours}h of data at {step} intervals...")
        print(f"Making 6 API calls (instead of ~2400 sequential calls)")
        print()

        # Fetch each metric type (6 total API calls)
        for metric_name, config in self.BULK_QUERIES.items():
            print(f"  [{list(self.BULK_QUERIES.keys()).index(metric_name) + 1}/6] Fetching {metric_name}...", end=" ")

            result = self._safe_fetch(
                metric_name,
                config['query'],
                hours,
                step
            )

            if result and result.get("status") == "success":
                parsed = self._parse_bulk_response(result, config['label_key'])

                # Normalize symbol format (BTCUSDT -> BTC) if needed
                if config['normalize'] == 'futures':
                    parsed = self._normalize_futures_to_spot(parsed)

                self._raw_data[metric_name] = parsed
                print(f"{len(parsed)} assets")
            else:
                self._raw_data[metric_name] = {}
                print("FAILED")

        print()

        # Build per-asset data structures
        return self._build_all_asset_data()

    def _safe_fetch(self, metric_name: str, query: str, hours: int, step: str) -> Optional[dict]:
        """Fetch with error handling."""
        try:
            return self.api.query_range(query, hours, step)
        except Exception as e:
            self.errors.append({
                'metric': metric_name,
                'error': str(e),
                'error_type': type(e).__name__
            })
            return None

    def _parse_bulk_response(self, result: dict, label_key: str) -> Dict[str, pd.DataFrame]:
        """Parse bulk response into per-asset DataFrames."""
        if result.get("status") != "success":
            return {}

        data = result.get("data", {})
        if data.get("resultType") != "matrix":
            return {}

        data_by_asset = {}

        for series in data.get("result", []):
            # Extract asset identifier from metric labels
            asset_id = series.get("metric", {}).get(label_key)
            if not asset_id:
                continue

            # Convert values to DataFrame
            values = series.get("values", [])
            if not values:
                continue

            df = pd.DataFrame(values, columns=["timestamp", "value"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.set_index("timestamp", inplace=True)

            data_by_asset[asset_id] = df

        return data_by_asset

    def _normalize_futures_to_spot(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Convert BTCUSDT keys to BTC keys."""
        normalized = {}
        for symbol, df in data.items():
            # Remove USDT suffix if present
            if symbol.endswith('USDT'):
                asset = symbol[:-4]
            else:
                asset = symbol
            normalized[asset] = df
        return normalized

    def _build_all_asset_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Build complete data dict for all assets."""
        # Get set of all assets from price metric (most complete)
        if 'price' not in self._raw_data:
            return {}

        all_assets = sorted(self._raw_data['price'].keys())
        result = {}
        skipped = 0

        for asset in all_assets:
            asset_data = self._build_single_asset_data(asset)
            if asset_data:
                result[asset] = asset_data
            else:
                skipped += 1

        print(f"Built data for {len(result)} assets ({skipped} skipped due to missing required data)")

        return result

    def _build_single_asset_data(self, asset: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Build data dict for one asset in DataFetcher-compatible format.

        Returns dict with keys: price, total_borrow, total_repay, rsi,
        funding_rate, open_interest, _asset
        """
        required = ['price', 'total_borrow', 'total_repay']

        # Check required metrics exist
        for metric in required:
            if asset not in self._raw_data.get(metric, {}):
                return None

        data = {}

        # Required metrics - rename column to metric name
        for metric in required:
            df = self._raw_data[metric][asset].copy()
            df.columns = [metric]
            data[metric] = df

        # Get reference index from price for alignment
        ref_index = data['price'].index

        # Optional metrics with NaN fallback
        for metric in ['rsi', 'funding_rate', 'open_interest']:
            if asset in self._raw_data.get(metric, {}):
                df = self._raw_data[metric][asset].copy()
                df.columns = [metric]
                # Reindex to match price timestamps
                data[metric] = df.reindex(ref_index, method='nearest',
                                          tolerance=pd.Timedelta('5m'))
            else:
                # Create NaN-filled DataFrame with matching index
                data[metric] = pd.DataFrame({metric: np.nan}, index=ref_index)

        # Store asset identifier
        data['_asset'] = asset

        return data

    def export_csv(self, output_dir: str = "output") -> str:
        """
        Export all data to CSV files.

        Structure:
            output/bulk_data_YYYY-MM-DD_HHMMSS/
                metadata.csv
                price/BTC.csv, ETH.csv, ...
                total_borrow/BTC.csv, ...
                ...
        """
        if not self.fetch_time:
            raise ValueError("No data fetched yet. Call fetch_all_metrics() first.")

        timestamp = self.fetch_time.strftime('%Y-%m-%d_%H%M%S')
        output_path = Path(output_dir) / f"bulk_data_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Metadata
        metadata = pd.DataFrame([
            {'parameter': 'fetch_time', 'value': str(self.fetch_time)},
            {'parameter': 'hours', 'value': self.hours},
            {'parameter': 'step', 'value': self.step},
            {'parameter': 'total_assets', 'value': len(self._raw_data.get('price', {}))},
            {'parameter': 'api_calls', 'value': 6},
        ])
        metadata.to_csv(output_path / 'metadata.csv', index=False)

        # Assets list
        assets = sorted(self._raw_data.get('price', {}).keys())
        pd.DataFrame({'asset': assets}).to_csv(output_path / 'assets.csv', index=False)

        # Per-metric directories
        total_files = 0
        for metric_name, asset_data in self._raw_data.items():
            metric_dir = output_path / metric_name
            metric_dir.mkdir(exist_ok=True)

            for asset, df in asset_data.items():
                df.to_csv(metric_dir / f"{asset}.csv")
                total_files += 1

        print(f"Exported {total_files} CSV files to: {output_path}")

        return str(output_path)

    def get_available_assets(self) -> List[str]:
        """Get list of assets with price data."""
        if 'price' not in self._raw_data:
            return []
        return sorted(self._raw_data['price'].keys())

    def get_complete_assets(self) -> List[str]:
        """Get list of assets with all required metrics."""
        if 'price' not in self._raw_data:
            return []

        complete = []
        for asset in self._raw_data['price'].keys():
            has_all = all(
                asset in self._raw_data.get(m, {})
                for m in ['price', 'total_borrow', 'total_repay']
            )
            if has_all:
                complete.append(asset)

        return sorted(complete)

    def summary(self) -> str:
        """Return a summary of fetched data."""
        lines = [
            f"Bulk Data Fetch Summary",
            f"=======================",
            f"Fetch time: {self.fetch_time}",
            f"Data range: {self.hours} hours at {self.step} step",
            f"API calls: 6 (vs ~{len(self.get_available_assets()) * 6} sequential)",
            f"",
            f"Assets by metric:",
        ]

        for metric in self.BULK_QUERIES.keys():
            count = len(self._raw_data.get(metric, {}))
            lines.append(f"  {metric}: {count}")

        lines.extend([
            f"",
            f"Total available: {len(self.get_available_assets())} assets",
            f"Complete (all required): {len(self.get_complete_assets())} assets",
        ])

        if self.errors:
            lines.extend([
                f"",
                f"Errors ({len(self.errors)}):",
            ])
            for err in self.errors:
                lines.append(f"  {err['metric']}: {err['error']}")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Bulk fetch all CalmCrypto data in minimal API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch 7 days of data, export to CSV
    python bulk_data_fetcher.py --hours 168 --export-csv

    # Fetch 24 hours with 1-minute resolution
    python bulk_data_fetcher.py --hours 24 --step 1m

    # Custom output directory
    python bulk_data_fetcher.py --hours 168 --export-csv --output-dir ./my_data
        """
    )
    parser.add_argument('--hours', type=int, default=168,
                        help='Hours of historical data (default: 168 = 7 days)')
    parser.add_argument('--step', type=str, default='5m',
                        help='Data interval (default: 5m)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for CSV export (default: output)')
    parser.add_argument('--export-csv', action='store_true',
                        help='Export data to CSV files')
    parser.add_argument('--list-assets', action='store_true',
                        help='List all available assets after fetching')

    args = parser.parse_args()

    # Fetch data
    fetcher = BulkDataFetcher()
    all_data = fetcher.fetch_all_metrics(hours=args.hours, step=args.step)

    # Print summary
    print()
    print(fetcher.summary())

    # List assets if requested
    if args.list_assets:
        print()
        print("Available assets:")
        assets = fetcher.get_available_assets()
        # Print in columns
        cols = 8
        for i in range(0, len(assets), cols):
            row = assets[i:i+cols]
            print("  " + ", ".join(row))

    # Export if requested
    if args.export_csv:
        print()
        fetcher.export_csv(args.output_dir)

    return all_data


if __name__ == "__main__":
    main()
