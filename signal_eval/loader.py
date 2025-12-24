"""Load evaluation results from existing CSV output."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


class ResultsLoader:
    """Load evaluation results from CSV files."""

    def __init__(self, output_dir: str):
        """
        Initialize loader with output directory path.

        Args:
            output_dir: Path to output directory containing CSVs
        """
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        self.summary_path = self.output_dir / "summary.csv"
        if not self.summary_path.exists():
            raise FileNotFoundError(f"summary.csv not found in {output_dir}")

    def load_summary(self) -> pd.DataFrame:
        """Load summary rankings."""
        df = pd.read_csv(self.summary_path, index_col=0)
        return df

    def load_signal_data(self, signal_name: str) -> Optional[pd.DataFrame]:
        """Load signal data CSV."""
        data_path = self.output_dir / f"{signal_name}_data.csv"
        if data_path.exists():
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
        return None

    def load_signal_metrics(self, signal_name: str) -> Optional[Dict[str, Any]]:
        """Load signal metrics CSV as dict."""
        metrics_path = self.output_dir / f"{signal_name}_metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            metrics = {}
            for _, row in df.iterrows():
                key = row['metric']
                value = row['value']
                # Convert string booleans
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                else:
                    # Try to convert to float
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass
                metrics[key] = value
            return metrics
        return None

    def load_rolling_data(self, signal_name: str) -> Optional[pd.DataFrame]:
        """Load rolling signal quality data."""
        rolling_path = self.output_dir / f"{signal_name}_rolling.csv"
        if rolling_path.exists():
            return pd.read_csv(rolling_path, index_col=0, parse_dates=True)
        return None

    def get_available_signals(self) -> list:
        """Get list of signals available in output directory."""
        signals = []
        for f in self.output_dir.glob("*_data.csv"):
            signal_name = f.stem.replace("_data", "")
            signals.append(signal_name)
        return signals

    def load_all(self) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        Load all data from output directory.

        Returns:
            Tuple of (rankings DataFrame, results dict)
        """
        rankings = self.load_summary()
        results = {}

        for signal_name in self.get_available_signals():
            result = {
                'signal_name': signal_name,
                'metrics': self.load_signal_metrics(signal_name) or {},
            }

            # Load rolling data if available
            rolling_df = self.load_rolling_data(signal_name)
            if rolling_df is not None:
                result['rolling_data'] = rolling_df
            else:
                result['rolling_data'] = pd.DataFrame()

            # Get composite score from rankings
            if signal_name in rankings['signal_name'].values:
                row = rankings[rankings['signal_name'] == signal_name].iloc[0]
                result['composite_score'] = row.get('composite_score', 0)
            else:
                result['composite_score'] = result['metrics'].get('composite_score', 0)

            results[signal_name] = result

        return rankings, results


class MockEvaluator:
    """Mock evaluator that holds loaded results for report generation."""

    def __init__(self, rankings: pd.DataFrame, results: Dict[str, Dict]):
        """
        Initialize with loaded data.

        Args:
            rankings: Summary rankings DataFrame
            results: Dict of signal_name -> result dict
        """
        self._rankings = rankings
        self.results = results

    def evaluate_all(self) -> pd.DataFrame:
        """Return loaded rankings."""
        return self._rankings


def load_from_output(output_dir: str) -> MockEvaluator:
    """
    Load evaluation results and create mock evaluator for report generation.

    Args:
        output_dir: Path to output directory with CSVs

    Returns:
        MockEvaluator instance ready for report generation
    """
    loader = ResultsLoader(output_dir)
    rankings, results = loader.load_all()
    return MockEvaluator(rankings, results)
