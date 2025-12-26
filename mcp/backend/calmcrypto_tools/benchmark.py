"""benchmark_all_assets MCP tool."""

import logging
import uuid
import warnings
import sys
import io
from pathlib import Path

import pandas as pd

# Add parent project to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from request_logger import log_request
from mcp_service import format_csv_response

logger = logging.getLogger(__name__)

# Suppress warnings during batch processing
warnings.filterwarnings('ignore')


def _benchmark_single_asset(asset: str, config, demo: bool = False) -> dict:
    """
    Run benchmark for a single asset, return summary metrics.

    Returns None if benchmark fails.
    """
    try:
        from signal_eval.data_fetcher import DataFetcher
        from signal_eval.signals import SignalRegistry
        from signal_eval.evaluator import SignalEvaluator

        # Fetch data
        fetcher = DataFetcher(demo=demo, asset=asset)
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
            return None

        # Extract summary metrics
        top_row = rankings.iloc[0]
        return {
            'asset': asset,
            'best_composite_score': top_row['composite_score'],
            'best_signal': top_row['signal_name'],
            'avg_composite_score': rankings['composite_score'].mean(),
            'significant_signals': int(rankings['granger_significant'].sum()),
            'best_effective_hit_rate': rankings['effective_hit_rate'].max(),
            'best_ic': rankings['best_spearman_ic'].abs().max(),
            'signals_evaluated': len(rankings),
        }
    except Exception as e:
        logger.debug(f"Benchmark failed for {asset}: {e}")
        return None


def register_benchmark_all_assets(local_mcp_instance, csv_dir, requests_dir):
    """Register the benchmark_all_assets tool."""

    @local_mcp_instance.tool()
    def benchmark_all_assets(
        requester: str,
        days: int = 7,
        top_n_assets: int = 0,
        demo: bool = False
    ) -> str:
        """
        Evaluate all assets by signal predictability and rank them.

        Runs signal evaluation on multiple assets and returns a ranking
        of assets by how predictable their price movements are based on
        trading signals.

        Parameters:
            requester (str): Identifier of who is calling this tool (e.g., 'trading-agent', 'user-alex').
                Used for request logging and audit purposes.
            days (int): Days of historical data to analyze (default: 7).
            top_n_assets (int): Limit to first N assets alphabetically (0 = all, default: 0).
            demo (bool): Use demo data instead of live API (default: False).

        Returns:
            str: Formatted CSV response with asset rankings by predictability.

        CSV Output Columns:
            - asset (string): Asset symbol
            - best_composite_score (float): Highest signal predictability score (0-1)
            - best_signal (string): Name of best-performing signal
            - avg_composite_score (float): Average score across all signals
            - significant_signals (integer): Count of statistically significant signals
            - best_effective_hit_rate (float): Best directional accuracy
            - best_ic (float): Best Information Coefficient
            - signals_evaluated (integer): Total signals tested

        Example usage:
            benchmark_all_assets(requester="my-agent", days=7)
            benchmark_all_assets(requester="my-agent", days=14, top_n_assets=50)

        Use Cases:
            - Find most predictable assets for trading
            - Compare signal effectiveness across the market
            - Identify assets with strong trading signals
            - Prioritize which assets to focus trading strategies on

        Note:
            - This tool can take several minutes to run for all assets
            - Use top_n_assets to limit scope for faster results
            - Assets with insufficient data will be skipped
        """
        logger.info(f"benchmark_all_assets invoked by {requester}: days={days}, top_n_assets={top_n_assets}")

        try:
            from dashboard import CalmCryptoAPI
            from signal_eval.config import Config

            config = Config()
            config.data_hours = days * 24

            api = CalmCryptoAPI()
            assets = api.get_all_assets()
            assets.sort()

            if top_n_assets > 0:
                assets = assets[:top_n_assets]

            logger.info(f"Benchmarking {len(assets)} assets...")

            results = []
            failed = []

            for i, asset in enumerate(assets, 1):
                result = _benchmark_single_asset(asset, config, demo=demo)

                if result:
                    results.append(result)
                    logger.info(f"[{i}/{len(assets)}] {asset}: score={result['best_composite_score']:.3f}")
                else:
                    failed.append(asset)
                    logger.info(f"[{i}/{len(assets)}] {asset}: FAILED")

            if not results:
                error_msg = "No results. All assets failed benchmark."
                log_request(
                    requests_dir=requests_dir,
                    requester=requester,
                    tool_name="benchmark_all_assets",
                    input_params={"days": days, "top_n_assets": top_n_assets, "demo": demo},
                    output_result=error_msg
                )
                return error_msg

            df = pd.DataFrame(results)
            df = df.sort_values('best_composite_score', ascending=False)
            df = df.reset_index(drop=True)

            filename = f"asset_benchmark_{str(uuid.uuid4())[:8]}.csv"
            filepath = csv_dir / filename
            df.to_csv(filepath, index=False)

            result = format_csv_response(filepath, df)

            # Add summary statistics
            result += f"\n\nSummary:\n"
            result += f"- Assets analyzed: {len(results)}\n"
            result += f"- Assets failed: {len(failed)}\n"
            result += f"- Top asset: {df.iloc[0]['asset']} (score: {df.iloc[0]['best_composite_score']:.3f})\n"

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="benchmark_all_assets",
                input_params={"days": days, "top_n_assets": top_n_assets, "demo": demo},
                output_result=result
            )

            return result

        except Exception as e:
            error_msg = f"Error running benchmark: {str(e)}"
            logger.error(error_msg)

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="benchmark_all_assets",
                input_params={"days": days, "top_n_assets": top_n_assets, "demo": demo},
                output_result=error_msg
            )

            return error_msg
