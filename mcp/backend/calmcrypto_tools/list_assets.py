"""list_assets MCP tool."""

import logging
import uuid
import pandas as pd
import sys
from pathlib import Path

# Add parent project to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from request_logger import log_request
from mcp_service import format_csv_response

logger = logging.getLogger(__name__)


def register_list_assets(local_mcp_instance, csv_dir, requests_dir):
    """Register the list_assets tool."""

    @local_mcp_instance.tool()
    def list_assets(requester: str) -> str:
        """
        Get all available cryptocurrency assets from CalmCrypto API.

        Returns a CSV file with all tradable asset symbols available in the
        CalmCrypto data feed (BTC, ETH, SOL, and 400+ altcoins).

        Parameters:
            requester (str): Identifier of who is calling this tool (e.g., 'trading-agent', 'user-alex').
                Used for request logging and audit purposes.

        Returns:
            str: Formatted response with CSV file info, schema, sample data, and Python snippet.

        CSV Output Columns:
            - asset (string): Asset symbol (e.g., 'BTC', 'ETH', 'SOL')

        Example usage:
            list_assets(requester="my-agent")

        Use Cases:
            - Get available assets before running benchmarks
            - Check if a specific asset is supported
            - Iterate through all assets for batch processing
        """
        logger.info(f"list_assets invoked by {requester}")

        try:
            # Import here to avoid import errors at module load time
            from dashboard import CalmCryptoAPI

            api = CalmCryptoAPI()
            assets = api.get_all_assets()
            assets.sort()

            df = pd.DataFrame({"asset": assets})

            filename = f"available_assets_{str(uuid.uuid4())[:8]}.csv"
            filepath = csv_dir / filename
            df.to_csv(filepath, index=False)

            result = format_csv_response(filepath, df)

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="list_assets",
                input_params={},
                output_result=result
            )

            return result

        except Exception as e:
            logger.error(f"Error in list_assets: {e}")
            error_msg = f"Error fetching assets: {str(e)}"

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="list_assets",
                input_params={},
                output_result=error_msg
            )

            return error_msg
