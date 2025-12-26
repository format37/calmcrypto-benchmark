#!/usr/bin/env python
"""Benchmark all available assets and rank by signal predictability."""

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard import CalmCryptoAPI
from signal_eval.data_fetcher import DataFetcher
from signal_eval.signals import SignalRegistry
from signal_eval.evaluator import SignalEvaluator
from signal_eval.config import Config

# Suppress warnings during batch processing
warnings.filterwarnings('ignore')


def benchmark_asset(asset: str, config: Config, demo: bool = False) -> dict:
    """
    Run benchmark for a single asset, return summary metrics.

    Returns None if benchmark fails.
    """
    try:
        # Fetch data
        fetcher = DataFetcher(demo=demo, asset=asset)
        raw_data = fetcher.fetch_all(hours=config.data_hours, step=config.step)

        # Build signals
        registry = SignalRegistry.from_raw_data(raw_data)
        price = registry.get_price_series(raw_data)

        # Evaluate (suppress per-signal errors in batch mode)
        import io
        import sys
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
        return None


def generate_html_report(results_df: pd.DataFrame, output_path: Path) -> str:
    """Generate interactive HTML report for multi-asset benchmark."""

    # Sort by best composite score
    df = results_df.sort_values('best_composite_score', ascending=False)
    top_n = min(50, len(df))  # Show top 50 in charts

    # Chart 1: Asset rankings bar chart
    fig_rankings = go.Figure()
    chart_df = df.head(top_n).sort_values('best_composite_score', ascending=True)

    colors = [f'rgba(88, 166, 255, {0.4 + 0.6 * (i / len(chart_df))})'
              for i in range(len(chart_df))]

    fig_rankings.add_trace(go.Bar(
        y=chart_df['asset'],
        x=chart_df['best_composite_score'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.3f}" for x in chart_df['best_composite_score']],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.4f}<br>"
            "<extra></extra>"
        )
    ))

    fig_rankings.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=100, t=20, b=20),
        xaxis_title="Best Composite Score",
        height=max(400, top_n * 20),
    )

    # Chart 2: Best signal distribution
    signal_counts = df['best_signal'].value_counts().head(15)
    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=signal_counts.index,
        y=signal_counts.values,
        marker_color='#58a6ff',
    ))
    fig_signals.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=20, b=100),
        xaxis_title="Signal",
        yaxis_title="# Assets where this is best signal",
        xaxis=dict(tickangle=45),
    )

    # Chart 3: Score vs IC scatter
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df['best_ic'],
        y=df['best_composite_score'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['significant_signals'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Sig. Signals')
        ),
        text=df['asset'],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "IC: %{x:.3f}<br>"
            "Score: %{y:.3f}<br>"
            "<extra></extra>"
        )
    ))
    fig_scatter.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=20, b=60),
        xaxis_title="Best Information Coefficient",
        yaxis_title="Best Composite Score",
    )

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Asset Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #c9d1d9;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #30363d;
            margin-bottom: 30px;
        }}
        h1 {{ color: #58a6ff; font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ color: #8b949e; font-size: 1.1em; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #21262d;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border: 1px solid #30363d;
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #58a6ff; }}
        .stat-label {{ color: #8b949e; margin-top: 5px; }}
        .top-asset {{ color: #3fb950; }}
        .card {{
            background: #21262d;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #30363d;
            margin-bottom: 20px;
        }}
        .card h2 {{
            color: #58a6ff;
            font-size: 1.3em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #30363d;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }}
        th {{ color: #58a6ff; }}
        tr:hover {{ background: rgba(88, 166, 255, 0.1); }}
        footer {{
            text-align: center;
            padding: 30px 0;
            color: #8b949e;
            border-top: 1px solid #30363d;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Multi-Asset Benchmark Report</h1>
            <p class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Assets Analyzed: {len(df)}</p>
        </header>

        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value top-asset">{df.iloc[0]['asset']}</div>
                <div class="stat-label">Most Predictable Asset</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{df.iloc[0]['best_composite_score']:.3f}</div>
                <div class="stat-label">Best Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{df['best_composite_score'].mean():.3f}</div>
                <div class="stat-label">Average Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(df)}</div>
                <div class="stat-label">Assets Analyzed</div>
            </div>
        </div>

        <div class="card">
            <h2>Top 20 Most Predictable Assets</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Asset</th>
                    <th>Best Score</th>
                    <th>Best Signal</th>
                    <th>Avg Score</th>
                    <th>Sig. Signals</th>
                    <th>Best Hit Rate</th>
                    <th>Best IC</th>
                </tr>
                {"".join(f'''
                <tr>
                    <td>{i+1}</td>
                    <td><b>{row['asset']}</b></td>
                    <td>{row['best_composite_score']:.3f}</td>
                    <td>{row['best_signal']}</td>
                    <td>{row['avg_composite_score']:.3f}</td>
                    <td>{int(row['significant_signals'])}</td>
                    <td>{row['best_effective_hit_rate']:.1%}</td>
                    <td>{row['best_ic']:.3f}</td>
                </tr>
                ''' for i, row in df.head(20).iterrows())}
            </table>
        </div>

        <div class="card">
            <h2>Asset Rankings by Best Composite Score (Top {top_n})</h2>
            <div id="rankings"></div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Most Common Best Signals</h2>
                <div id="signals"></div>
            </div>
            <div class="card">
                <h2>Score vs Information Coefficient</h2>
                <div id="scatter"></div>
            </div>
        </div>

        <footer>
            <p>CalmCrypto Multi-Asset Benchmark System</p>
        </footer>
    </div>

    <script>
        var rankingsData = {fig_rankings.to_json()};
        Plotly.newPlot('rankings', rankingsData.data, rankingsData.layout, {{responsive: true}});

        var signalsData = {fig_signals.to_json()};
        Plotly.newPlot('signals', signalsData.data, signalsData.layout, {{responsive: true}});

        var scatterData = {fig_scatter.to_json()};
        Plotly.newPlot('scatter', scatterData.data, scatterData.layout, {{responsive: true}});
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all assets and rank by signal predictability"
    )
    parser.add_argument('--demo', action='store_true', help='Use demo data')
    parser.add_argument('--days', type=int, default=7, help='Days of data (default: 7)')
    parser.add_argument('--top-n-assets', type=int, default=None,
                        help='Limit to first N assets')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    args = parser.parse_args()

    # Setup
    config = Config()
    config.data_hours = args.days * 24
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get assets
    print("Fetching available assets...")
    api = CalmCryptoAPI()
    assets = api.get_all_assets()
    assets.sort()

    if args.top_n_assets:
        assets = assets[:args.top_n_assets]

    print(f"Benchmarking {len(assets)} assets...")
    print("-" * 60)

    # Benchmark each asset
    results = []
    failed = []

    for i, asset in enumerate(assets, 1):
        result = benchmark_asset(asset, config, demo=args.demo)

        if result:
            results.append(result)
            print(f"[{i}/{len(assets)}] {asset}: score={result['best_composite_score']:.3f}, "
                  f"best={result['best_signal']}")
        else:
            failed.append(asset)
            print(f"[{i}/{len(assets)}] {asset}: FAILED")

    print("-" * 60)
    print(f"Complete. {len(results)} succeeded, {len(failed)} failed.")

    if not results:
        print("No results to save.")
        return

    # Create DataFrame and save CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_composite_score', ascending=False)
    results_df = results_df.reset_index(drop=True)

    csv_path = output_dir / "asset_benchmark_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Generate HTML report
    html_path = output_dir / "multi_asset_benchmark_report.html"
    generate_html_report(results_df, html_path)
    print(f"Saved: {html_path}")

    # Print top 10
    print("\n" + "=" * 60)
    print("TOP 10 MOST PREDICTABLE ASSETS")
    print("=" * 60)
    for i, row in results_df.head(10).iterrows():
        print(f"{i+1:2}. {row['asset']:8} score={row['best_composite_score']:.3f}  "
              f"best_signal={row['best_signal']}")

    if failed:
        print(f"\nFailed assets ({len(failed)}): {', '.join(failed[:10])}"
              + ("..." if len(failed) > 10 else ""))


if __name__ == "__main__":
    main()
