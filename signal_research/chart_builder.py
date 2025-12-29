"""Build interactive multi-subplot Plotly charts for signal research.

Creates charts with:
- Top panel: Price line with signal markers (triangles)
- Lower panels: Raw data (RSI, borrow, repay, funding, OI)
- Interactive legend with Show All / Remove All buttons
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color palette for signals (distinct colors for up to 15 signals)
SIGNAL_COLORS = [
    '#00ff88',  # bright green
    '#ff6b6b',  # coral red
    '#4ecdc4',  # teal
    '#ffe66d',  # yellow
    '#95e1d3',  # mint
    '#f38181',  # salmon
    '#aa96da',  # lavender
    '#fcbad3',  # pink
    '#a8d8ea',  # light blue
    '#f8b500',  # orange
    '#7fcd91',  # sage
    '#d4a5a5',  # dusty rose
    '#79d70f',  # lime
    '#ffd3b6',  # peach
    '#98ddca',  # aqua
]


class ResearchChartBuilder:
    """Build interactive research charts with signal overlays."""

    def __init__(self, asset: str, timeframe: str = '5m'):
        """
        Initialize chart builder.

        Args:
            asset: Asset symbol (e.g., 'BTC')
            timeframe: Data interval (e.g., '5m', '15m')
        """
        self.asset = asset
        self.timeframe = timeframe
        self.fig = None

    def build_chart(
        self,
        price: pd.Series,
        raw_data: Dict[str, pd.DataFrame],
        signal_events: Dict[str, pd.DataFrame],
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Build complete research chart with price, signals, and raw data panels.

        Args:
            price: Price series with datetime index
            raw_data: Dict with keys: rsi, total_borrow, total_repay, funding_rate, open_interest
            signal_events: Dict mapping signal_name -> events DataFrame

        Returns:
            Plotly Figure object
        """
        # Create subplot layout: price + 5 raw data panels
        self.fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.40, 0.12, 0.12, 0.12, 0.12, 0.12],
            subplot_titles=[
                f'{self.asset} Price with Signals',
                'RSI',
                'Borrow / Repay Volume',
                'Funding Rate',
                'Open Interest',
                'Net Flow (Borrow - Repay)'
            ]
        )

        # Row 1: Price with signal markers
        self._add_price_trace(price, row=1)
        self._add_signal_markers(price, signal_events, row=1)

        # Row 2: RSI
        if 'rsi' in raw_data:
            rsi = raw_data['rsi']
            if isinstance(rsi, pd.DataFrame):
                rsi = rsi.iloc[:, 0] if len(rsi.columns) > 0 else pd.Series()
            self._add_raw_data_panel(rsi, 'RSI', row=2, color='#9b59b6')
            # Add overbought/oversold lines
            self.fig.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.5, row=2, col=1)
            self.fig.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.5, row=2, col=1)

        # Row 3: Borrow / Repay
        if 'total_borrow' in raw_data and 'total_repay' in raw_data:
            borrow = raw_data['total_borrow']
            repay = raw_data['total_repay']
            if isinstance(borrow, pd.DataFrame):
                borrow = borrow.iloc[:, 0] if len(borrow.columns) > 0 else pd.Series()
            if isinstance(repay, pd.DataFrame):
                repay = repay.iloc[:, 0] if len(repay.columns) > 0 else pd.Series()
            self._add_borrow_repay_panel(borrow, repay, row=3)

        # Row 4: Funding Rate
        if 'funding_rate' in raw_data:
            funding = raw_data['funding_rate']
            if isinstance(funding, pd.DataFrame):
                funding = funding.iloc[:, 0] if len(funding.columns) > 0 else pd.Series()
            self._add_raw_data_panel(funding, 'Funding Rate', row=4, color='#e74c3c')
            # Add zero line
            self.fig.add_hline(y=0, line_dash='solid', line_color='white', opacity=0.3, row=4, col=1)

        # Row 5: Open Interest
        if 'open_interest' in raw_data:
            oi = raw_data['open_interest']
            if isinstance(oi, pd.DataFrame):
                oi = oi.iloc[:, 0] if len(oi.columns) > 0 else pd.Series()
            self._add_raw_data_panel(oi, 'Open Interest', row=5, color='#3498db')

        # Row 6: Net Flow
        if 'total_borrow' in raw_data and 'total_repay' in raw_data:
            borrow = raw_data['total_borrow']
            repay = raw_data['total_repay']
            if isinstance(borrow, pd.DataFrame):
                borrow = borrow.iloc[:, 0]
            if isinstance(repay, pd.DataFrame):
                repay = repay.iloc[:, 0]
            net_flow = borrow - repay
            self._add_net_flow_panel(net_flow, row=6)

        # Update layout
        chart_title = title or f'{self.asset} Signal Research ({self.timeframe})'
        self._apply_layout(chart_title)
        self._add_legend_buttons()

        return self.fig

    def _add_price_trace(self, price: pd.Series, row: int):
        """Add price line to specified row."""
        self.fig.add_trace(
            go.Scatter(
                x=price.index,
                y=price.values,
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=1.5),
                legendgroup='price',
                hovertemplate='<b>Price</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
            ),
            row=row, col=1
        )

    def _add_signal_markers(
        self,
        price: pd.Series,
        signal_events: Dict[str, pd.DataFrame],
        row: int
    ):
        """Add UP/DOWN triangle markers for all signals."""
        color_idx = 0

        for signal_name, events in signal_events.items():
            if events.empty:
                continue

            color = SIGNAL_COLORS[color_idx % len(SIGNAL_COLORS)]
            color_idx += 1

            # UP signals (green triangles)
            up_events = events[events['direction'] == 'UP']
            if not up_events.empty:
                # Find price at event timestamps
                up_prices = []
                up_times = []
                up_reasons = []
                up_conf = []

                for _, event in up_events.iterrows():
                    ts = event['timestamp']
                    if ts in price.index:
                        up_times.append(ts)
                        up_prices.append(price.loc[ts])
                        up_reasons.append(event.get('reason', ''))
                        up_conf.append(event.get('confidence', 0))
                    elif len(price.index) > 0:
                        # Find nearest timestamp
                        idx = price.index.get_indexer([ts], method='nearest')[0]
                        if 0 <= idx < len(price):
                            up_times.append(price.index[idx])
                            up_prices.append(price.iloc[idx])
                            up_reasons.append(event.get('reason', ''))
                            up_conf.append(event.get('confidence', 0))

                if up_times:
                    self.fig.add_trace(
                        go.Scatter(
                            x=up_times,
                            y=up_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                color='green',
                                size=10,
                                line=dict(color='darkgreen', width=1)
                            ),
                            name=f'{signal_name} (Long)',
                            legendgroup=signal_name,
                            showlegend=True,
                            customdata=list(zip(up_conf, up_reasons)),
                            hovertemplate=(
                                f'<b>{signal_name} - LONG</b><br>'
                                'Time: %{x}<br>'
                                'Price: $%{y:,.2f}<br>'
                                'Confidence: %{customdata[0]:.0%}<br>'
                                'Reason: %{customdata[1]}<br>'
                                '<extra></extra>'
                            )
                        ),
                        row=row, col=1
                    )

            # DOWN signals (red triangles)
            down_events = events[events['direction'] == 'DOWN']
            if not down_events.empty:
                down_prices = []
                down_times = []
                down_reasons = []
                down_conf = []

                for _, event in down_events.iterrows():
                    ts = event['timestamp']
                    if ts in price.index:
                        down_times.append(ts)
                        down_prices.append(price.loc[ts])
                        down_reasons.append(event.get('reason', ''))
                        down_conf.append(event.get('confidence', 0))
                    elif len(price.index) > 0:
                        idx = price.index.get_indexer([ts], method='nearest')[0]
                        if 0 <= idx < len(price):
                            down_times.append(price.index[idx])
                            down_prices.append(price.iloc[idx])
                            down_reasons.append(event.get('reason', ''))
                            down_conf.append(event.get('confidence', 0))

                if down_times:
                    self.fig.add_trace(
                        go.Scatter(
                            x=down_times,
                            y=down_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                color='red',
                                size=10,
                                line=dict(color='darkred', width=1)
                            ),
                            name=f'{signal_name} (Short)',
                            legendgroup=signal_name,
                            showlegend=True,
                            customdata=list(zip(down_conf, down_reasons)),
                            hovertemplate=(
                                f'<b>{signal_name} - SHORT</b><br>'
                                'Time: %{x}<br>'
                                'Price: $%{y:,.2f}<br>'
                                'Confidence: %{customdata[0]:.0%}<br>'
                                'Reason: %{customdata[1]}<br>'
                                '<extra></extra>'
                            )
                        ),
                        row=row, col=1
                    )

    def _add_raw_data_panel(
        self,
        data: pd.Series,
        name: str,
        row: int,
        color: str
    ):
        """Add raw data line to subplot."""
        if data.empty:
            return

        self.fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                mode='lines',
                name=name,
                line=dict(color=color, width=1),
                legendgroup=f'raw_{name.lower().replace(" ", "_")}',
                hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:,.4f}}<extra></extra>'
            ),
            row=row, col=1
        )

    def _add_borrow_repay_panel(
        self,
        borrow: pd.Series,
        repay: pd.Series,
        row: int
    ):
        """Add borrow and repay volume as overlapping lines."""
        if not borrow.empty:
            self.fig.add_trace(
                go.Scatter(
                    x=borrow.index,
                    y=borrow.values,
                    mode='lines',
                    name='Borrow',
                    line=dict(color='#27ae60', width=1),
                    legendgroup='borrow_repay',
                    hovertemplate='<b>Borrow</b><br>%{x}<br>$%{y:,.0f}<extra></extra>'
                ),
                row=row, col=1
            )

        if not repay.empty:
            self.fig.add_trace(
                go.Scatter(
                    x=repay.index,
                    y=repay.values,
                    mode='lines',
                    name='Repay',
                    line=dict(color='#c0392b', width=1),
                    legendgroup='borrow_repay',
                    hovertemplate='<b>Repay</b><br>%{x}<br>$%{y:,.0f}<extra></extra>'
                ),
                row=row, col=1
            )

    def _add_net_flow_panel(self, net_flow: pd.Series, row: int):
        """Add net flow (borrow - repay) with positive/negative coloring."""
        if net_flow.empty:
            return

        # Create color array based on positive/negative
        colors = ['#27ae60' if v >= 0 else '#c0392b' for v in net_flow.values]

        self.fig.add_trace(
            go.Bar(
                x=net_flow.index,
                y=net_flow.values,
                name='Net Flow',
                marker_color=colors,
                legendgroup='net_flow',
                hovertemplate='<b>Net Flow</b><br>%{x}<br>$%{y:,.0f}<extra></extra>'
            ),
            row=row, col=1
        )

        # Add zero line
        self.fig.add_hline(y=0, line_dash='solid', line_color='white', opacity=0.3, row=row, col=1)

    def _apply_layout(self, title: str):
        """Apply dark theme layout and styling."""
        self.fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#58a6ff'),
                x=0.5
            ),
            template='plotly_dark',
            paper_bgcolor='#0d1117',
            plot_bgcolor='#161b22',
            font=dict(color='#c9d1d9'),
            hovermode='x unified',
            height=1200,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(22,27,34,0.8)',
                bordercolor='#30363d',
                borderwidth=1,
                itemclick='toggle',
                itemdoubleclick='toggleothers'
            ),
            margin=dict(l=60, r=20, t=100, b=40)
        )

        # Update y-axis labels
        self.fig.update_yaxes(title_text='Price (USDT)', row=1, col=1)
        self.fig.update_yaxes(title_text='RSI', row=2, col=1)
        self.fig.update_yaxes(title_text='Volume (USDT)', row=3, col=1)
        self.fig.update_yaxes(title_text='Rate', row=4, col=1)
        self.fig.update_yaxes(title_text='OI (USDT)', row=5, col=1)
        self.fig.update_yaxes(title_text='Net Flow', row=6, col=1)

        # Update x-axis for bottom subplot
        self.fig.update_xaxes(title_text='Date', row=6, col=1)

        # Apply grid styling to all subplots
        for i in range(1, 7):
            self.fig.update_xaxes(
                gridcolor='#30363d',
                gridwidth=0.5,
                row=i, col=1
            )
            self.fig.update_yaxes(
                gridcolor='#30363d',
                gridwidth=0.5,
                row=i, col=1
            )

    def _add_legend_buttons(self):
        """Add Show All / Remove All buttons."""
        self.fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    direction='right',
                    x=0.0,
                    y=1.12,
                    xanchor='left',
                    yanchor='top',
                    buttons=[
                        dict(
                            label='Show All',
                            method='restyle',
                            args=[{'visible': True}]
                        ),
                        dict(
                            label='Hide All',
                            method='restyle',
                            args=[{'visible': 'legendonly'}]
                        )
                    ],
                    showactive=False,
                    bgcolor='#21262d',
                    bordercolor='#30363d',
                    font=dict(color='#c9d1d9')
                )
            ]
        )

    def save_html(self, output_path: str) -> str:
        """
        Save chart as HTML file.

        Args:
            output_path: Output file path

        Returns:
            Path to saved file
        """
        if self.fig is None:
            raise ValueError("No chart built. Call build_chart() first.")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.fig.write_html(output_path)
        return output_path


def build_research_chart(
    asset: str,
    price: pd.Series,
    raw_data: Dict[str, pd.DataFrame],
    signal_events: Dict[str, pd.DataFrame],
    timeframe: str = '5m',
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Convenience function to build and optionally save a research chart.

    Args:
        asset: Asset symbol
        price: Price series
        raw_data: Raw metric DataFrames
        signal_events: Signal events by signal name
        timeframe: Data interval
        output_path: Optional path to save HTML

    Returns:
        Plotly Figure
    """
    builder = ResearchChartBuilder(asset, timeframe)
    fig = builder.build_chart(price, raw_data, signal_events)

    if output_path:
        builder.save_html(output_path)

    return fig
