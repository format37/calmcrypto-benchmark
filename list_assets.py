#!/usr/bin/env python
"""List all available assets from the Grafana API."""

from pathlib import Path
from datetime import datetime

import pandas as pd

from dashboard import CalmCryptoAPI


def main():
    api = CalmCryptoAPI()
    assets = api.get_all_assets()
    assets.sort()

    print(f"Available assets ({len(assets)}):")
    for asset in assets:
        print(f"  {asset}")

    # Save to CSV
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    df = pd.DataFrame({"asset": assets})
    output_path = output_dir / "available_assets.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
