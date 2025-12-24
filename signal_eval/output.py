"""CSV output handling for signal evaluation results."""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd


class OutputManager:
    """Manage CSV output for signal evaluation."""

    def __init__(self, base_dir: str = "output"):
        """
        Initialize output manager.

        Args:
            base_dir: Base output directory
        """
        self.base_dir = Path(base_dir)
        self.run_dir: Optional[Path] = None
        self.saved_files: List[str] = []

    def create_run_directory(self) -> Path:
        """Create timestamped directory for this evaluation run."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir = self.base_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_signal_data(
        self,
        signal_name: str,
        data_df: pd.DataFrame
    ) -> str:
        """
        Save signal data to CSV.

        Args:
            signal_name: Name of the signal
            data_df: DataFrame with signal values, price, and forward returns

        Returns:
            Path to saved file
        """
        if self.run_dir is None:
            self.create_run_directory()

        filename = f"{signal_name}_data.csv"
        filepath = self.run_dir / filename
        data_df.to_csv(filepath)
        self.saved_files.append(str(filepath))
        return str(filepath)

    def save_signal_metrics(
        self,
        signal_name: str,
        metrics: Dict[str, Any],
        by_period: Optional[Dict[str, Dict]] = None
    ) -> str:
        """
        Save signal metrics to CSV.

        Args:
            signal_name: Name of the signal
            metrics: Dict of metric name -> value
            by_period: Optional per-period metrics

        Returns:
            Path to saved file
        """
        if self.run_dir is None:
            self.create_run_directory()

        # Build rows
        rows = []

        # Main metrics
        metric_descriptions = {
            'best_pearson_ic': 'Pearson Information Coefficient (best period)',
            'best_spearman_ic': 'Spearman Information Coefficient (best period)',
            'best_ic_ir': 'IC Information Ratio (mean/std, best period)',
            'best_lag': 'Best predictive lag (in 5min periods)',
            'lead_lag_correlation': 'Correlation at best lag',
            'lead_lag_score': 'Lead-lag score (0-1, higher = indicator leads)',
            'hit_rate': 'Overall directional accuracy (raw)',
            'hit_rate_bullish': 'Hit rate when signal is bullish',
            'hit_rate_bearish': 'Hit rate when signal is bearish',
            'total_signals': 'Total number of signal changes evaluated',
            'is_contrarian': 'True if signal predicts opposite direction (hit_rate < 50%)',
            'effective_hit_rate': 'Effective accuracy: max(hit_rate, 1-hit_rate)',
            'granger_best_lag': 'Best lag for Granger causality',
            'granger_p_value': 'Granger test p-value (lower = more significant)',
            'granger_significant': 'Granger test significant at 0.05 level',
            'granger_score': 'Granger score (0-1)',
            'current_ic': 'Current rolling IC value',
            'current_hit_rate': 'Current rolling hit rate',
            'current_score': 'Current rolling composite score',
            'is_improving': 'Signal quality is improving',
        }

        for key, value in metrics.items():
            rows.append({
                'metric': key,
                'value': value,
                'description': metric_descriptions.get(key, '')
            })

        # Per-period metrics
        if by_period:
            for period_label, period_metrics in by_period.items():
                for key, value in period_metrics.items():
                    if key != 'period':
                        rows.append({
                            'metric': f"{key}_{period_label}",
                            'value': value,
                            'description': f"{key} for {period_label} forward"
                        })

        df = pd.DataFrame(rows)
        filename = f"{signal_name}_metrics.csv"
        filepath = self.run_dir / filename
        df.to_csv(filepath, index=False)
        self.saved_files.append(str(filepath))
        return str(filepath)

    def save_summary(self, rankings_df: pd.DataFrame) -> str:
        """
        Save summary rankings to CSV.

        Args:
            rankings_df: DataFrame with all signals ranked

        Returns:
            Path to saved file
        """
        if self.run_dir is None:
            self.create_run_directory()

        filename = "summary.csv"
        filepath = self.run_dir / filename
        rankings_df.to_csv(filepath)
        self.saved_files.append(str(filepath))
        return str(filepath)

    def save_rolling_data(
        self,
        signal_name: str,
        rolling_df: pd.DataFrame
    ) -> str:
        """
        Save rolling signal quality data to CSV.

        Args:
            signal_name: Name of the signal
            rolling_df: DataFrame from rolling_signal_quality()

        Returns:
            Path to saved file
        """
        if self.run_dir is None:
            self.create_run_directory()

        filename = f"{signal_name}_rolling.csv"
        filepath = self.run_dir / filename
        rolling_df.to_csv(filepath)
        self.saved_files.append(str(filepath))
        return str(filepath)

    def save_metadata(
        self,
        asset: str,
        data_hours: int,
        step: str,
        signals_evaluated: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """
        Save run metadata to CSV for LLM agent consumption.

        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            data_hours: Hours of data analyzed
            step: Data step interval (e.g., "5m")
            signals_evaluated: Number of signals evaluated
            start_time: Data start time (optional)
            end_time: Data end time (optional)

        Returns:
            Path to saved file
        """
        if self.run_dir is None:
            self.create_run_directory()

        rows = [
            {'parameter': 'asset', 'value': asset},
            {'parameter': 'data_hours', 'value': data_hours},
            {'parameter': 'step', 'value': step},
            {'parameter': 'signals_evaluated', 'value': signals_evaluated},
            {'parameter': 'generation_time', 'value': datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        ]

        if start_time:
            rows.append({'parameter': 'start_time', 'value': start_time.strftime("%Y-%m-%d %H:%M:%S")})
        if end_time:
            rows.append({'parameter': 'end_time', 'value': end_time.strftime("%Y-%m-%d %H:%M:%S")})

        df = pd.DataFrame(rows)
        filename = "metadata.csv"
        filepath = self.run_dir / filename
        df.to_csv(filepath, index=False)
        self.saved_files.append(str(filepath))
        return str(filepath)

    def get_output_summary(self) -> Dict[str, Any]:
        """Get summary of saved files."""
        return {
            'run_directory': str(self.run_dir) if self.run_dir else None,
            'files_saved': len(self.saved_files),
            'file_list': self.saved_files.copy()
        }


def save_evaluation_results(
    evaluator,
    output_dir: str = "output",
    top_n: Optional[int] = None,
    include_rolling: bool = False,
    asset: Optional[str] = None,
    data_hours: Optional[int] = None,
    step: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save all evaluation results to CSV files.

    Args:
        evaluator: SignalEvaluator instance (after evaluate_all())
        output_dir: Base output directory
        top_n: Number of top signals to save (None = all)
        include_rolling: Whether to save rolling data CSVs
        asset: Asset symbol for metadata (e.g., "BTC")
        data_hours: Hours of data for metadata
        step: Data step interval for metadata

    Returns:
        Dict with output summary
    """
    output = OutputManager(output_dir)
    output.create_run_directory()

    # Get rankings
    rankings = evaluator.evaluate_all()
    if top_n:
        rankings = rankings.head(top_n)

    # Save metadata first (if asset provided)
    if asset:
        output.save_metadata(
            asset=asset,
            data_hours=data_hours or 168,
            step=step or "5m",
            signals_evaluated=len(rankings)
        )

    # Save summary
    output.save_summary(rankings)

    # Save individual signal files
    for signal_name in rankings['signal_name']:
        result = evaluator.results.get(signal_name)
        if not result:
            continue

        # Save data
        data_df = evaluator.get_signal_data(signal_name)
        output.save_signal_data(signal_name, data_df)

        # Save metrics
        output.save_signal_metrics(
            signal_name,
            result['metrics'],
            result.get('by_period')
        )

        # Save rolling data if requested
        if include_rolling and 'rolling_data' in result:
            rolling_df = result['rolling_data']
            if not rolling_df.empty:
                output.save_rolling_data(signal_name, rolling_df)

    return output.get_output_summary()
