"""predict_price MCP tool."""

import logging
import uuid
import json
import sys
from pathlib import Path

# Add parent project to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from request_logger import log_request

logger = logging.getLogger(__name__)


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

        Uses top-ranked signals to generate UP/DOWN predictions for
        multiple timeframes (1h, 12h, 24h) with confidence levels.

        Parameters:
            requester (str): Identifier of who is calling this tool (e.g., 'trading-agent', 'user-alex').
                Used for request logging and audit purposes.
            asset (str): Asset symbol (e.g., 'BTC', 'ETH', 'SOL').
            top_n (int): Number of top signals to use for prediction (default: 5).
            days (int): Days of historical data for signal evaluation (default: 14).

        Returns:
            str: Formatted prediction summary with JSON file path.

        Output includes:
            - current_price: Current asset price
            - predictions: Dict with 1h/12h/24h predictions
                - direction: 'UP' or 'DOWN'
                - probability: 0.0-1.0
                - confidence: 'Very Low' to 'Very High'
            - signals_used: List of contributing signals with details

        Example usage:
            predict_price(requester="my-agent", asset="BTC")
            predict_price(requester="my-agent", asset="ETH", top_n=10, days=7)

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

            # Save JSON to data directory
            filename = f"prediction_{result['asset']}_{str(uuid.uuid4())[:8]}.json"
            filepath = csv_dir / filename
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)

            # Build human-readable response
            preds = result['predictions']
            response = f"""Prediction for {result['asset']}

File: {filename}
Current Price: ${result['current_price']:,.4f}
Generated: {result['timestamp'][:19]}

Predictions:
| Timeframe | Direction | Probability | Confidence |
|-----------|-----------|-------------|------------|
| 1h        | {preds['1h']['direction']:<9} | {preds['1h']['probability']:>6.1%}      | {preds['1h']['confidence']:<12} |
| 12h       | {preds['12h']['direction']:<9} | {preds['12h']['probability']:>6.1%}      | {preds['12h']['confidence']:<12} |
| 24h       | {preds['24h']['direction']:<9} | {preds['24h']['probability']:>6.1%}      | {preds['24h']['confidence']:<12} |

Top Contributing Signals:
"""
            for i, sig in enumerate(result['signals_used'][:5], 1):
                contr = " (contrarian)" if sig['is_contrarian'] else ""
                response += f"  {i}. {sig['name']} ({sig['direction']}) -> {sig['prediction']} ({sig['hit_rate']:.0%} hit rate{contr})\n"

            response += f"""
Python snippet to load:
```python
import json
with open('data/mcp-calmcrypto/{filename}') as f:
    prediction = json.load(f)
print(prediction['predictions'])
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
