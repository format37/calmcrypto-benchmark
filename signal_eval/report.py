"""Interactive report generation for signal evaluation."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def generate_report(
    evaluator,
    output_dir: str = "output",
    top_n: Optional[int] = None,
    asset: Optional[str] = None
) -> str:
    """
    Generate interactive HTML report comparing signal benchmarks.

    Args:
        evaluator: SignalEvaluator instance (after evaluate_all())
        output_dir: Output directory
        top_n: Number of top signals to include
        asset: Asset symbol for report title (e.g., "BTC", "ETH")

    Returns:
        Path to generated HTML report
    """
    rankings = evaluator.evaluate_all()
    if top_n:
        rankings = rankings.head(top_n)

    # Generate tab identifier: asset name or random 4-digit number
    tab_id = asset.upper() if asset else f"{random.randint(1000, 9999)}"

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report_dir = Path(output_dir) / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate report HTML
    html_content = _build_report_html(evaluator, rankings, tab_id)

    # Save report
    report_path = report_dir / "signal_benchmark_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)

    return str(report_path)


def _build_report_html(evaluator, rankings: pd.DataFrame, tab_id: str) -> str:
    """Build the complete HTML report with all visualizations."""

    # Create individual figures
    fig_rankings = _create_rankings_chart(rankings)
    fig_radar = _create_radar_chart(rankings)
    fig_heatmap = _create_metrics_heatmap(rankings)
    fig_scatter = _create_ic_vs_hitrate_scatter(rankings)
    fig_lead_lag = _create_lead_lag_chart(rankings)
    fig_rolling = _create_rolling_comparison(evaluator, rankings)

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{tab_id} Signal Benchmark</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #c9d1d9;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #30363d;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #58a6ff;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #8b949e;
            font-size: 1.1em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        .grid-full {{
            grid-column: span 2;
        }}
        .card {{
            background: #21262d;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #30363d;
        }}
        .card h2 {{
            color: #58a6ff;
            font-size: 1.3em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #30363d;
        }}
        .chart {{
            width: 100%;
            height: 400px;
        }}
        .chart-large {{
            height: 500px;
        }}
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
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #58a6ff;
        }}
        .stat-label {{
            color: #8b949e;
            margin-top: 5px;
        }}
        .top-signal {{
            color: #3fb950;
        }}
        footer {{
            text-align: center;
            padding: 30px 0;
            color: #8b949e;
            border-top: 1px solid #30363d;
            margin-top: 30px;
        }}
        @media (max-width: 1200px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            .grid-full {{
                grid-column: span 1;
            }}
            .summary-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Signal Benchmark Report</h1>
            <p class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Signals Evaluated: {len(rankings)}</p>
        </header>

        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value top-signal">{rankings.iloc[0]['signal_name']}</div>
                <div class="stat-label">Top Signal</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{rankings.iloc[0]['composite_score']:.3f}</div>
                <div class="stat-label">Best Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{abs(rankings['best_spearman_ic'].max()):.3f}</div>
                <div class="stat-label">Best IC</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{rankings['hit_rate'].max():.1%}</div>
                <div class="stat-label">Best Hit Rate</div>
            </div>
        </div>

        <div class="grid">
            <div class="card grid-full">
                <h2>Signal Rankings by Composite Score</h2>
                <div id="rankings" class="chart chart-large"></div>
            </div>

            <div class="card">
                <h2>Metrics Radar Comparison</h2>
                <div id="radar" class="chart"></div>
            </div>

            <div class="card">
                <h2>IC vs Hit Rate</h2>
                <div id="scatter" class="chart"></div>
            </div>

            <div class="card grid-full">
                <h2>Metrics Heatmap</h2>
                <div id="heatmap" class="chart chart-large"></div>
            </div>

            <div class="card grid-full">
                <h2>Lead-Lag Analysis</h2>
                <div id="leadlag" class="chart"></div>
            </div>

            <div class="card grid-full">
                <h2>Rolling Signal Quality (Top 5)</h2>
                <div id="rolling" class="chart chart-large"></div>
            </div>
        </div>

        <footer>
            <p>CalmCrypto Signal Evaluation System</p>
        </footer>
    </div>

    <script>
        // Rankings Chart
        var rankingsData = {fig_rankings.to_json()};
        Plotly.newPlot('rankings', rankingsData.data, rankingsData.layout, {{responsive: true}});

        // Radar Chart
        var radarData = {fig_radar.to_json()};
        Plotly.newPlot('radar', radarData.data, radarData.layout, {{responsive: true}});

        // Scatter Chart
        var scatterData = {fig_scatter.to_json()};
        Plotly.newPlot('scatter', scatterData.data, scatterData.layout, {{responsive: true}});

        // Heatmap
        var heatmapData = {fig_heatmap.to_json()};
        Plotly.newPlot('heatmap', heatmapData.data, heatmapData.layout, {{responsive: true}});

        // Lead-Lag Chart
        var leadlagData = {fig_lead_lag.to_json()};
        Plotly.newPlot('leadlag', leadlagData.data, leadlagData.layout, {{responsive: true}});

        // Rolling Chart
        var rollingData = {fig_rolling.to_json()};
        Plotly.newPlot('rolling', rollingData.data, rollingData.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    return html


def _create_rankings_chart(rankings: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of signal rankings."""
    # Sort by composite score ascending for horizontal bars (bottom to top)
    df = rankings.sort_values('composite_score', ascending=True)

    # Color gradient based on score
    colors = [f'rgba(88, 166, 255, {0.4 + 0.6 * (i / len(df))})' for i in range(len(df))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['signal_name'],
        x=df['composite_score'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.3f}" for x in df['composite_score']],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.4f}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=100, t=20, b=20),
        xaxis_title="Composite Score",
        yaxis_title="",
        showlegend=False,
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    )

    return fig


def _create_radar_chart(rankings: pd.DataFrame) -> go.Figure:
    """Create radar chart comparing top signals across metrics."""
    top_n = min(5, len(rankings))
    top_signals = rankings.head(top_n)

    categories = ['IC', 'IC-IR', 'Eff. Hit Rate', 'Lead-Lag', 'Granger']

    fig = go.Figure()

    # Define colors directly as hex
    color_palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

    for idx, (_, row) in enumerate(top_signals.iterrows()):
        # Normalize metrics to 0-1 scale
        ic_val = row.get('best_spearman_ic', 0)
        ic_ir_val = row.get('best_ic_ir', 0)
        hr_val = row.get('hit_rate', 0.5)
        ll_val = row.get('lead_lag_score', 0)
        gr_val = row.get('granger_score', 0)

        # Handle NaN values
        if pd.isna(ic_val):
            ic_val = 0
        if pd.isna(ic_ir_val):
            ic_ir_val = 0
        if pd.isna(hr_val):
            hr_val = 0.5
        if pd.isna(ll_val):
            ll_val = 0
        if pd.isna(gr_val):
            gr_val = 0

        # Use effective hit rate (handles contrarian signals)
        effective_hr = max(hr_val, 1 - hr_val)

        values = [
            min(abs(ic_val) / 0.1, 1),           # IC
            min(max(ic_ir_val, 0) / 2, 1),       # IC-IR
            (effective_hr - 0.5) * 2,            # Effective Hit Rate (0.5->0, 1.0->1)
            ll_val,                              # Lead-Lag
            gr_val,                              # Granger
        ]
        values.append(values[0])  # Close the polygon

        color = color_palette[idx % len(color_palette)]
        # Convert hex to rgba for fill
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba({r},{g},{b},0.2)',
            line=dict(color=color, width=2),
            name=row['signal_name']
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(255,255,255,0.1)'
            ),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=60, t=40, b=60),
    )

    return fig


def _create_metrics_heatmap(rankings: pd.DataFrame) -> go.Figure:
    """Create heatmap of all metrics across signals."""
    # Select metrics to display
    metrics = [
        'composite_score', 'best_spearman_ic', 'best_ic_ir',
        'hit_rate', 'lead_lag_score', 'granger_score'
    ]

    display_names = [
        'Composite', 'Spearman IC', 'IC-IR',
        'Hit Rate', 'Lead-Lag', 'Granger'
    ]

    # Build matrix
    data = []
    for metric in metrics:
        if metric in rankings.columns:
            values = rankings[metric].fillna(0).values
            # Normalize
            if values.max() != values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = values
            data.append(normalized)
        else:
            data.append([0] * len(rankings))

    z = np.array(data)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=rankings['signal_name'].values,
        y=display_names,
        colorscale='Viridis',
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{y}: %{z:.3f}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=100, r=20, t=20, b=100),
        xaxis=dict(tickangle=45),
    )

    return fig


def _create_ic_vs_hitrate_scatter(rankings: pd.DataFrame) -> go.Figure:
    """Create scatter plot of IC vs Hit Rate."""
    fig = go.Figure()

    # Size based on composite score
    sizes = (rankings['composite_score'] - rankings['composite_score'].min())
    sizes = (sizes / sizes.max() * 30 + 10) if sizes.max() > 0 else [20] * len(rankings)

    fig.add_trace(go.Scatter(
        x=rankings['best_spearman_ic'].abs(),
        y=rankings['hit_rate'],
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=rankings['composite_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Score')
        ),
        text=rankings['signal_name'],
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "IC: %{x:.4f}<br>"
            "Hit Rate: %{y:.2%}<br>"
            "<extra></extra>"
        )
    ))

    # Add reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="50% (no edge)")

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=40, b=60),
        xaxis_title="Information Coefficient (abs)",
        yaxis_title="Raw Hit Rate (below 50% = contrarian)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickformat='.0%'),
    )

    return fig


def _create_lead_lag_chart(rankings: pd.DataFrame) -> go.Figure:
    """Create chart showing lead-lag characteristics."""
    fig = go.Figure()

    # Sort by best_lag
    df = rankings.sort_values('best_lag', ascending=True)

    colors = ['#3fb950' if lag > 0 else '#f85149' if lag < 0 else '#8b949e'
              for lag in df['best_lag']]

    fig.add_trace(go.Bar(
        x=df['signal_name'],
        y=df['best_lag'],
        marker_color=colors,
        text=[f"{int(lag)} ({lag*5}min)" for lag in df['best_lag']],
        textposition='outside',
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Best Lag: %{y} periods<br>"
            "(%{text})<br>"
            "<extra></extra>"
        )
    ))

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)")

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=40, b=100),
        xaxis_title="",
        yaxis_title="Lead-Lag (periods)",
        xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        showlegend=False,
        annotations=[
            dict(
                x=0.02, y=0.98, xref='paper', yref='paper',
                text='<b>Green</b> = Signal leads price<br><b>Red</b> = Price leads signal',
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(0,0,0,0.5)',
                borderpad=4
            )
        ]
    )

    return fig


def _create_rolling_comparison(evaluator, rankings: pd.DataFrame) -> go.Figure:
    """Create rolling signal quality comparison for top signals."""
    fig = go.Figure()

    top_n = min(5, len(rankings))
    color_palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

    for idx, (_, row) in enumerate(rankings.head(top_n).iterrows()):
        signal_name = row['signal_name']
        result = evaluator.results.get(signal_name)

        if result and 'rolling_data' in result:
            rolling_df = result['rolling_data']
            if not rolling_df.empty and 'signal_score' in rolling_df.columns:
                # Resample for smoother visualization
                score = rolling_df['signal_score'].dropna()
                if len(score) > 0:
                    fig.add_trace(go.Scatter(
                        x=score.index,
                        y=score.values,
                        mode='lines',
                        name=signal_name,
                        line=dict(color=color_palette[idx % len(color_palette)], width=2),
                        hovertemplate=(
                            f"<b>{signal_name}</b><br>"
                            "Time: %{x}<br>"
                            "Score: %{y:.4f}<br>"
                            "<extra></extra>"
                        )
                    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=40, b=60),
        xaxis_title="Time",
        yaxis_title="Rolling Signal Score",
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified'
    )

    return fig
