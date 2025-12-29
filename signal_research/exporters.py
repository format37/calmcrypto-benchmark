"""Export functions for raw data CSV and consolidated XLS.

Provides:
- Per-asset CSV export for raw metrics and signal events
- Consolidated XLS workbook with all assets
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np


def export_raw_data_csv(
    raw_data: Dict[str, pd.DataFrame],
    output_dir: Path,
    asset: str
) -> List[str]:
    """
    Export raw data for one asset to CSV files.

    Args:
        raw_data: Dict with keys: price, rsi, total_borrow, total_repay, funding_rate, open_interest
        output_dir: Base output directory
        asset: Asset symbol (creates subfolder)

    Returns:
        List of created file paths
    """
    asset_dir = output_dir / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    metric_names = ['price', 'rsi', 'total_borrow', 'total_repay', 'funding_rate', 'open_interest']

    for metric in metric_names:
        if metric not in raw_data:
            continue

        data = raw_data[metric]
        if isinstance(data, pd.DataFrame):
            # Flatten to single column if needed
            if len(data.columns) == 1:
                df = data.copy()
                df.columns = ['value']
            else:
                df = data.copy()
        elif isinstance(data, pd.Series):
            df = pd.DataFrame({'value': data})
        else:
            continue

        # Reset index to get timestamp column
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'timestamp'})

        # Save CSV
        filepath = asset_dir / f'{metric}.csv'
        df.to_csv(filepath, index=False)
        created_files.append(str(filepath))

    return created_files


def export_signal_events_csv(
    events: Dict[str, pd.DataFrame],
    output_dir: Path,
    asset: str
) -> str:
    """
    Export signal events to CSV.

    Args:
        events: Dict mapping signal_name -> events DataFrame
        output_dir: Base output directory
        asset: Asset symbol

    Returns:
        Path to created file
    """
    asset_dir = output_dir / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Merge all events
    all_dfs = [df for df in events.values() if not df.empty]

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged = merged.sort_values('timestamp').reset_index(drop=True)
    else:
        merged = pd.DataFrame(columns=['timestamp', 'signal_name', 'direction', 'confidence', 'reason'])

    filepath = asset_dir / 'signal_events.csv'
    merged.to_csv(filepath, index=False)
    return str(filepath)


def export_consolidated_xls(
    all_assets_data: Dict[str, Dict[str, Any]],
    output_path: Path,
    include_raw: bool = True
) -> str:
    """
    Export single XLS workbook with all data.

    Args:
        all_assets_data: Dict mapping asset -> {
            'raw_data': Dict of metric DataFrames,
            'signal_events': Dict of events DataFrames,
            'summary': Dict with summary stats
        }
        output_path: Full path to output XLS file
        include_raw: Whether to include raw data sheets

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary
        summary_rows = []
        for asset, data in all_assets_data.items():
            summary = data.get('summary', {})
            events = data.get('signal_events', {})

            # Count events
            total_up = 0
            total_down = 0
            most_active_signal = ''
            most_active_count = 0

            for signal_name, events_df in events.items():
                if events_df.empty:
                    continue
                up_count = len(events_df[events_df['direction'] == 'UP'])
                down_count = len(events_df[events_df['direction'] == 'DOWN'])
                total_up += up_count
                total_down += down_count

                signal_total = up_count + down_count
                if signal_total > most_active_count:
                    most_active_count = signal_total
                    most_active_signal = signal_name

            summary_rows.append({
                'asset': asset,
                'total_up_signals': total_up,
                'total_down_signals': total_down,
                'total_signals': total_up + total_down,
                'most_active_signal': most_active_signal,
                'date_range_start': summary.get('date_range_start', ''),
                'date_range_end': summary.get('date_range_end', ''),
            })

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('total_signals', ascending=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Per-asset sheets
        for asset, data in all_assets_data.items():
            # Signal events sheet
            events = data.get('signal_events', {})
            all_events_dfs = [df for df in events.values() if not df.empty]
            if all_events_dfs:
                merged_events = pd.concat(all_events_dfs, ignore_index=True)
                merged_events = merged_events.sort_values('timestamp').reset_index(drop=True)
                sheet_name = f'{asset}_signals'[:31]  # Excel sheet name limit
                merged_events.to_excel(writer, sheet_name=sheet_name, index=False)

            # Raw data sheet (merged)
            if include_raw:
                raw_data = data.get('raw_data', {})
                merged_raw = _merge_raw_data(raw_data)
                if not merged_raw.empty:
                    sheet_name = f'{asset}_raw'[:31]
                    merged_raw.to_excel(writer, sheet_name=sheet_name, index=False)

    return str(output_path)


def _merge_raw_data(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all raw metric DataFrames into single DataFrame."""
    metrics = ['price', 'rsi', 'total_borrow', 'total_repay', 'funding_rate', 'open_interest']

    dfs = []
    for metric in metrics:
        if metric not in raw_data:
            continue

        data = raw_data[metric]
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                df = data.copy()
                df.columns = [metric]
            elif len(data.columns) > 0:
                df = data.iloc[:, [0]].copy()
                df.columns = [metric]
            else:
                continue
        elif isinstance(data, pd.Series):
            df = pd.DataFrame({metric: data})
        else:
            continue

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Merge all on index
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how='outer')

    merged = merged.reset_index()
    if 'index' in merged.columns:
        merged = merged.rename(columns={'index': 'timestamp'})

    return merged


def create_output_directory(base_dir: str = 'output/research') -> Path:
    """
    Create timestamped output directory.

    Args:
        base_dir: Base output directory

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_metadata(
    output_dir: Path,
    assets: List[str],
    days: int,
    step: str,
    top_signals: int,
    timestamp: datetime
) -> str:
    """
    Export run metadata to CSV.

    Args:
        output_dir: Output directory
        assets: List of processed assets
        days: Days of data fetched
        step: Data interval
        top_signals: Number of signals used
        timestamp: Run timestamp

    Returns:
        Path to metadata file
    """
    metadata = pd.DataFrame([
        {'parameter': 'run_timestamp', 'value': str(timestamp)},
        {'parameter': 'days', 'value': days},
        {'parameter': 'step', 'value': step},
        {'parameter': 'top_signals', 'value': top_signals},
        {'parameter': 'assets_count', 'value': len(assets)},
        {'parameter': 'assets', 'value': ','.join(assets[:20]) + ('...' if len(assets) > 20 else '')},
    ])

    filepath = output_dir / 'metadata.csv'
    metadata.to_csv(filepath, index=False)
    return str(filepath)
