"""signal_eval MCP tool."""

import logging
import uuid
import warnings
import sys
import io
from pathlib import Path

# Add parent project to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from request_logger import log_request
from mcp_service import format_csv_response

logger = logging.getLogger(__name__)

# Suppress warnings during evaluation
warnings.filterwarnings('ignore')


def register_signal_eval(local_mcp_instance, csv_dir, requests_dir):
    """Register the signal_eval tool."""

    @local_mcp_instance.tool()
    def signal_eval(
        requester: str,
        asset: str = "BTC",
        days: int = 7,
        demo: bool = False,
        top_n: int = 10
    ) -> str:
        """
        Run deep signal evaluation for a single asset.

        Evaluates multiple trading signals against price movements using
        Information Coefficient, Granger causality, hit rate, and lead-lag analysis.
        Returns a ranked list of signals by predictive power.

        Parameters:
            requester (str): Identifier of who is calling this tool (e.g., 'trading-agent', 'user-alex').
                Used for request logging and audit purposes.
            asset (str): Asset symbol to analyze (e.g., 'BTC', 'ETH', 'SOL'). Default: 'BTC'.
            days (int): Days of historical data to analyze (default: 7).
            demo (bool): Use demo data instead of live API (default: False).
            top_n (int): Number of top signals to include in output (default: 10).

        Returns:
            str: Formatted CSV response with signal rankings.

        CSV Output Columns:
            - signal_name (string): Name of the signal
            - composite_score (float): Weighted predictive score (0-1, higher is better)
            - best_spearman_ic (float): Best Information Coefficient across periods
            - hit_rate (float): Raw directional accuracy
            - effective_hit_rate (float): Effective accuracy (handles contrarian signals)
            - is_contrarian (boolean): Whether signal predicts opposite direction
            - granger_significant (boolean): Statistical significance of predictive power
            - lead_lag_score (float): How well signal leads price movements

        Example usage:
            signal_eval(requester="my-agent", asset="BTC", days=7)
            signal_eval(requester="my-agent", asset="ETH", days=14, top_n=5)

        Use Cases:
            - Identify best predictive signals for a specific asset
            - Compare signal effectiveness across different assets
            - Find contrarian signals (where inverse logic applies)
            - Evaluate statistical significance of signal predictions
        """
        logger.info(f"signal_eval invoked by {requester}: asset={asset}, days={days}")

        try:
            # Import here to avoid import errors at module load time
            from signal_eval.config import Config
            from signal_eval.data_fetcher import DataFetcher
            from signal_eval.signals import SignalRegistry
            from signal_eval.evaluator import SignalEvaluator

            config = Config()
            config.data_hours = days * 24
            config.asset = asset.upper()
            config.top_n = top_n

            # Fetch data
            fetcher = DataFetcher(demo=demo, asset=config.asset)
            raw_data = fetcher.fetch_all(hours=config.data_hours, step=config.step)

            # Build signals
            registry = SignalRegistry.from_raw_data(raw_data)
            price = registry.get_price_series(raw_data)

            # Evaluate (suppress output)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                evaluator = SignalEvaluator(price, config)
                evaluator.add_signals(registry.all_signals())
                rankings = evaluator.evaluate_all()
            finally:
                sys.stdout = old_stdout

            if rankings.empty:
                error_msg = f"No signals could be evaluated for {asset}."
                log_request(
                    requests_dir=requests_dir,
                    requester=requester,
                    tool_name="signal_eval",
                    input_params={"asset": asset, "days": days, "demo": demo, "top_n": top_n},
                    output_result=error_msg
                )
                return error_msg

            # Save to CSV
            df = rankings.head(top_n)
            filename = f"signal_eval_{asset}_{str(uuid.uuid4())[:8]}.csv"
            filepath = csv_dir / filename
            df.to_csv(filepath, index=False)

            result = format_csv_response(filepath, df)

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="signal_eval",
                input_params={"asset": asset, "days": days, "demo": demo, "top_n": top_n},
                output_result=result
            )

            return result

        except Exception as e:
            error_msg = f"Error evaluating signals for {asset}: {str(e)}"
            logger.error(error_msg)

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="signal_eval",
                input_params={"asset": asset, "days": days, "demo": demo, "top_n": top_n},
                output_result=error_msg
            )

            return error_msg
