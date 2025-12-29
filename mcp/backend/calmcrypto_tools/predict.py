"""predict_price MCP tool."""

import logging
import uuid
import json
import csv
import sys
from pathlib import Path

# Add parent project to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from request_logger import log_request

logger = logging.getLogger(__name__)


def _save_prediction_csvs(result: dict, csv_dir: Path, prefix: str) -> tuple:
    """Save prediction results to CSV files.

    Returns:
        Tuple of (predictions_filepath, signals_filepath)
    """
    # Predictions CSV
    predictions_file = csv_dir / f"{prefix}_predictions.csv"
    with open(predictions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timeframe', 'direction', 'probability', 'confidence', 'confidence_score', 'signals_used', 'agreement_ratio'])
        for tf, pred in result['predictions'].items():
            writer.writerow([
                tf,
                pred['direction'],
                pred['probability'],
                pred['confidence'],
                pred.get('confidence_score', ''),
                pred.get('signals_used', ''),
                pred.get('agreement_ratio', '')
            ])

    # Signals CSV
    signals_file = csv_dir / f"{prefix}_signals.csv"
    signals = result.get('signals_analyzed', result.get('signals_used', []))
    with open(signals_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['signal', 'rule_type', 'interpretation', 'prediction', 'is_contrarian', 'current_ic', 'effective_hit_rate', 'best_lag'])
        for sig in signals:
            writer.writerow([
                sig.get('name', ''),
                sig.get('rule_type', ''),
                sig.get('interpretation', sig.get('direction', '')),
                sig.get('prediction', ''),
                sig.get('is_contrarian', False),
                sig.get('current_ic', sig.get('hit_rate', '')),
                sig.get('effective_hit_rate', sig.get('hit_rate', '')),
                sig.get('best_lag', '')
            ])

    return predictions_file, signals_file


def register_predict_price(local_mcp_instance, csv_dir, requests_dir):
    """Register the predict_price tool."""

    @local_mcp_instance.tool()
    def predict_price(
        requester: str,
        asset: str,
        top_n: int = 5,
        days: int = 14
    ) -> str:
        """
        Predict price direction with probability for an asset.

        Uses config-driven signal interpretation rules and timeframe-weighted
        aggregation to generate UP/DOWN predictions for 1h, 12h, 24h timeframes.

        Parameters:
            requester (str): Identifier of who is calling this tool (e.g., 'trading-agent', 'user-alex').
                Used for request logging and audit purposes.
            asset (str): Asset symbol (e.g., 'BTC', 'ETH', 'SOL').
            top_n (int): Number of top signals to use for prediction (default: 5).
            days (int): Days of historical data for signal evaluation (default: 14).

        Returns:
            str: Formatted prediction summary with file paths.

        Generated Files:
            - {prefix}_predictions.csv: Timeframe predictions (direction, probability, confidence, agreement)
            - {prefix}_signals.csv: Signal analysis (rule_type, interpretation, prediction, metrics)
            - {prefix}.json: Full prediction data

        CSV Schemas:
            predictions.csv columns:
                timeframe, direction, probability, confidence, confidence_score, signals_used, agreement_ratio
            signals.csv columns:
                signal, rule_type, interpretation, prediction, is_contrarian, current_ic, effective_hit_rate, best_lag

        Example usage:
            predict_price(requester="my-agent", asset="BTC")
            predict_price(requester="my-agent", asset="ETH", top_n=10, days=7)

        py_eval example:
            import pandas as pd
            preds = pd.read_csv('data/mcp-calmcrypto/prediction_BTC_abc12345_predictions.csv')
            sigs = pd.read_csv('data/mcp-calmcrypto/prediction_BTC_abc12345_signals.csv')
            print("=== Predictions ===")
            print(preds.to_string(index=False))
            print("\n=== Signals ===")
            print(sigs.to_string(index=False))

        Use Cases:
            - Get directional price prediction for trading decisions
            - Compare prediction confidence across multiple assets
            - Understand which signals are driving the prediction
        """
        logger.info(f"predict_price invoked by {requester}: asset={asset}")

        try:
            # Import here to avoid import errors at module load time
            from predict_price import predict_asset

            result = predict_asset(asset, top_n=top_n, days=days)

            # Generate unique prefix for files
            file_prefix = f"prediction_{result['asset']}_{str(uuid.uuid4())[:8]}"

            # Save JSON
            json_filename = f"{file_prefix}.json"
            json_filepath = csv_dir / json_filename
            with open(json_filepath, 'w') as f:
                json.dump(result, f, indent=2)

            # Save CSV tables
            predictions_file, signals_file = _save_prediction_csvs(result, csv_dir, file_prefix)

            # Build human-readable response
            preds = result['predictions']

            response = f"""Prediction for {result['asset']}

Current Price: ${result['current_price']:,.4f}
Generated: {result['timestamp'][:19]}

Predictions:
| Timeframe | Direction | Probability | Confidence   | Signals | Agreement |
|-----------|-----------|-------------|--------------|---------|-----------|
| 1h        | {preds['1h']['direction']:<9} | {preds['1h']['probability']:>6.1%}      | {preds['1h']['confidence']:<12} | {preds['1h'].get('signals_used', '-'):<7} | {preds['1h'].get('agreement_ratio', 0.5):>5.0%}      |
| 12h       | {preds['12h']['direction']:<9} | {preds['12h']['probability']:>6.1%}      | {preds['12h']['confidence']:<12} | {preds['12h'].get('signals_used', '-'):<7} | {preds['12h'].get('agreement_ratio', 0.5):>5.0%}      |
| 24h       | {preds['24h']['direction']:<9} | {preds['24h']['probability']:>6.1%}      | {preds['24h']['confidence']:<12} | {preds['24h'].get('signals_used', '-'):<7} | {preds['24h'].get('agreement_ratio', 0.5):>5.0%}      |

Signal Analysis:
"""
            # Handle both old (signals_used) and new (signals_analyzed) formats
            signals = result.get('signals_analyzed', result.get('signals_used', []))
            for i, sig in enumerate(signals[:5], 1):
                contr = " [contrarian]" if sig.get('is_contrarian', False) else ""
                interp = sig.get('interpretation', sig.get('direction', ''))
                rule = sig.get('rule_type', 'unknown')
                pred = sig.get('prediction', '')
                response += f"  {i}. {sig['name']} ({rule}): {interp} -> {pred}{contr}\n"

            response += f"""
Files:
  - {predictions_file.name}
  - {signals_file.name}
  - {json_filename}

Python snippet to load:
```python
import pandas as pd
predictions = pd.read_csv('data/mcp-calmcrypto/{predictions_file.name}')
signals = pd.read_csv('data/mcp-calmcrypto/{signals_file.name}')
```"""

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="predict_price",
                input_params={"asset": asset, "top_n": top_n, "days": days},
                output_result=response
            )

            return response

        except ValueError as e:
            error_msg = f"Error predicting {asset}: {str(e)}"
            logger.error(error_msg)

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="predict_price",
                input_params={"asset": asset, "top_n": top_n, "days": days},
                output_result=error_msg
            )

            return error_msg

        except Exception as e:
            error_msg = f"Unexpected error predicting {asset}: {str(e)}"
            logger.error(error_msg)

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="predict_price",
                input_params={"asset": asset, "top_n": top_n, "days": days},
                output_result=error_msg
            )

            return error_msg
