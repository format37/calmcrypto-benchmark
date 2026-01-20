"""market_briefing MCP tool.

Provides comprehensive market state data for an asset, returning factual market
data WITHOUT trading instructions or predictions. Pure informational briefing.
"""

import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from PIL import Image as PILImage

# Add parent project to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from request_logger import log_request
from mcp_image_utils import to_mcp_image

logger = logging.getLogger(__name__)

# Dark theme colors
COLORS = {
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'text': '#e0e0e0',
    'grid': '#2a3f5f',
    'price': '#00d9ff',
    'rsi': '#9b59b6',
    'borrow': '#27ae60',
    'repay': '#e74c3c',
    'funding_pos': '#00ff88',
    'funding_neg': '#ff6b6b',
    'oi': '#3498db',
    'ema_short': '#f39c12',
    'ema_long': '#9b59b6',
}


def _fetch_market_data(asset: str, lookback_hours: int = 168) -> Dict[str, pd.DataFrame]:
    """Fetch all market data for an asset.

    Args:
        asset: Asset symbol (e.g., 'BTC', 'ETH')
        lookback_hours: Hours of historical data

    Returns:
        Dict with DataFrames for each metric
    """
    from signal_eval.data_fetcher import DataFetcher

    fetcher = DataFetcher(demo=False, asset=asset)
    data = fetcher.fetch_all(hours=lookback_hours, step="5m")

    return data


def _resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute data to hourly using last value."""
    if df.empty:
        return df
    return df.resample('1h').last().dropna()


def _get_series(data: Dict, key: str) -> pd.Series:
    """Extract series from data dict, handling DataFrame wrapping."""
    if key not in data or key.startswith('_'):
        return pd.Series()
    val = data[key]
    if isinstance(val, pd.DataFrame):
        return val.iloc[:, 0] if len(val.columns) > 0 else pd.Series()
    return val


def _last_value(series: pd.Series, default: float = np.nan) -> float:
    """Get last valid value from series."""
    if series.empty:
        return default
    return series.dropna().iloc[-1] if not series.dropna().empty else default


def _prev_value(series: pd.Series, periods: int = 24, default: float = np.nan) -> float:
    """Get value from n periods ago."""
    if series.empty or len(series) <= periods:
        return default
    return series.iloc[-periods - 1] if not pd.isna(series.iloc[-periods - 1]) else default


def _pct_change(current: float, previous: float) -> float:
    """Calculate percentage change."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return 0.0
    return (current - previous) / previous * 100


def _fmt_price(val: float, decimals: int = 2) -> str:
    """Format price with commas."""
    if pd.isna(val):
        return "N/A"
    return f"${val:,.{decimals}f}"


def _fmt_pct(val: float) -> str:
    """Format percentage."""
    if pd.isna(val):
        return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


def _fmt_funding(val: float) -> str:
    """Format funding rate as percentage."""
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:.4f}%"


def _fmt_volume(val: float) -> str:
    """Format large volume values."""
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1e9:
        return f"${val / 1e9:.2f}B"
    elif abs(val) >= 1e6:
        return f"${val / 1e6:.2f}M"
    elif abs(val) >= 1e3:
        return f"${val / 1e3:.2f}K"
    return f"${val:.0f}"


def _calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _calculate_volatility(series: pd.Series, window: int = 24) -> float:
    """Calculate rolling volatility (standard deviation of returns)."""
    if series.empty or len(series) < window:
        return np.nan
    returns = series.pct_change().dropna()
    if len(returns) < window:
        return np.nan
    return returns.iloc[-window:].std() * 100


def generate_market_analysis(data: Dict, asset: str) -> str:
    """Generate markdown analysis sections.

    Returns factual market data sections WITHOUT trading instructions.
    """
    # Extract series
    price = _get_series(data, 'price')
    rsi = _get_series(data, 'rsi')
    funding = _get_series(data, 'funding_rate')
    oi = _get_series(data, 'open_interest')
    borrow = _get_series(data, 'total_borrow')
    repay = _get_series(data, 'total_repay')

    # Current values
    current_price = _last_value(price)
    current_rsi = _last_value(rsi)
    current_funding = _last_value(funding)
    current_oi = _last_value(oi)
    current_borrow = _last_value(borrow)
    current_repay = _last_value(repay)

    # Previous values (24 periods = 24 hours if hourly, or 2 hours if 5-min)
    prev_price_1h = _prev_value(price, 12)  # 1 hour ago (12 * 5min)
    prev_price_24h = _prev_value(price, 288)  # 24 hours ago (288 * 5min)
    prev_price_7d = _prev_value(price, 2016)  # 7 days ago (2016 * 5min)

    # Calculate changes
    change_1h = _pct_change(current_price, prev_price_1h)
    change_24h = _pct_change(current_price, prev_price_24h)
    change_7d = _pct_change(current_price, prev_price_7d)

    # Calculate EMAs
    ema_12 = _last_value(_calculate_ema(price, 12))
    ema_26 = _last_value(_calculate_ema(price, 26))

    # Volatility
    volatility_24h = _calculate_volatility(price, 288)

    # Net flow
    net_flow = current_borrow - current_repay if not pd.isna(current_borrow) and not pd.isna(current_repay) else np.nan

    # Build analysis sections
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    sections = []

    # Header
    sections.append(f"# {asset} Market Briefing")
    sections.append(f"*Generated: {timestamp}*\n")

    # Price Action Section
    sections.append("## Price Action")
    sections.append(f"- **Current Price:** {_fmt_price(current_price)}")
    sections.append(f"- **1H Change:** {_fmt_pct(change_1h)}")
    sections.append(f"- **24H Change:** {_fmt_pct(change_24h)}")
    sections.append(f"- **7D Change:** {_fmt_pct(change_7d)}")
    sections.append(f"- **RSI (3m):** {current_rsi:.1f}" if not pd.isna(current_rsi) else "- **RSI:** N/A")
    sections.append(f"- **24H Volatility:** {volatility_24h:.2f}%" if not pd.isna(volatility_24h) else "- **24H Volatility:** N/A")
    sections.append(f"- **EMA 12:** {_fmt_price(ema_12)}")
    sections.append(f"- **EMA 26:** {_fmt_price(ema_26)}")

    # Trend context
    if not pd.isna(ema_12) and not pd.isna(ema_26):
        if ema_12 > ema_26:
            sections.append("- **EMA Trend:** Short-term EMA above long-term (bullish structure)")
        else:
            sections.append("- **EMA Trend:** Short-term EMA below long-term (bearish structure)")

    sections.append("")

    # Derivatives & Funding Section
    sections.append("## Derivatives & Funding")
    sections.append(f"- **Current Funding Rate:** {_fmt_funding(current_funding)}")
    if not pd.isna(current_funding):
        if current_funding > 0.0001:
            sections.append("  - Longs paying shorts (crowded long)")
        elif current_funding < -0.0001:
            sections.append("  - Shorts paying longs (crowded short)")
        else:
            sections.append("  - Near neutral funding")
    sections.append("")

    # Open Interest Section
    sections.append("## Open Interest & Positioning")
    sections.append(f"- **Current Open Interest:** {_fmt_volume(current_oi)}")
    prev_oi_24h = _prev_value(oi, 288)
    oi_change = _pct_change(current_oi, prev_oi_24h)
    sections.append(f"- **24H OI Change:** {_fmt_pct(oi_change)}")
    sections.append("")

    # Margin & Capital Flow Section
    sections.append("## Margin & Capital Flow")
    sections.append(f"- **24H Total Borrow:** {_fmt_volume(current_borrow)}")
    sections.append(f"- **24H Total Repay:** {_fmt_volume(current_repay)}")
    sections.append(f"- **Net Flow:** {_fmt_volume(net_flow)}")
    if not pd.isna(net_flow):
        if net_flow > 0:
            sections.append("  - Positive net flow (more borrowing than repaying)")
        else:
            sections.append("  - Negative net flow (more repaying than borrowing)")
    sections.append("")

    # RSI Analysis
    sections.append("## Technical Indicators")
    if not pd.isna(current_rsi):
        sections.append(f"- **RSI:** {current_rsi:.1f}")
        if current_rsi >= 70:
            sections.append("  - Overbought territory (>70)")
        elif current_rsi <= 30:
            sections.append("  - Oversold territory (<30)")
        elif current_rsi >= 60:
            sections.append("  - Upper range (60-70)")
        elif current_rsi <= 40:
            sections.append("  - Lower range (30-40)")
        else:
            sections.append("  - Neutral range (40-60)")

    return "\n".join(sections)


def _apply_dark_style(ax, title: str = ""):
    """Apply dark theme styling to axis."""
    ax.set_facecolor(COLORS['panel'])
    ax.set_title(title, color=COLORS['text'], fontsize=12, pad=10)
    ax.tick_params(colors=COLORS['text'], labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.grid(True, alpha=0.3, color=COLORS['grid'])


def _fig_to_pil(fig) -> PILImage.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=COLORS['bg'],
                bbox_inches='tight', pad_inches=0.2)
    buf.seek(0)
    return PILImage.open(buf).convert('RGB')


def plot_price_action(data: Dict, asset: str) -> PILImage.Image:
    """Plot price with EMAs."""
    price = _get_series(data, 'price')
    if price.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    # Plot price
    ax.plot(price.index, price.values, color=COLORS['price'], linewidth=1.5, label='Price')

    # EMAs
    ema_12 = _calculate_ema(price, 12)
    ema_26 = _calculate_ema(price, 26)
    ax.plot(price.index, ema_12.values, color=COLORS['ema_short'], linewidth=1,
            linestyle='--', label='EMA 12', alpha=0.8)
    ax.plot(price.index, ema_26.values, color=COLORS['ema_long'], linewidth=1,
            linestyle='--', label='EMA 26', alpha=0.8)

    _apply_dark_style(ax, f"{asset} Price Action")
    ax.set_ylabel('Price (USDT)', color=COLORS['text'])
    ax.legend(loc='upper left', facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    plt.tight_layout()
    pil_img = _fig_to_pil(fig)
    plt.close(fig)

    return pil_img


def plot_rsi(data: Dict, asset: str) -> PILImage.Image:
    """Plot RSI with overbought/oversold zones."""
    rsi = _get_series(data, 'rsi')
    if rsi.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg'])

    ax.plot(rsi.index, rsi.values, color=COLORS['rsi'], linewidth=1.5)
    ax.axhline(y=70, color='#ff6b6b', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax.axhline(y=50, color=COLORS['grid'], linestyle='-', alpha=0.5)

    # Fill overbought/oversold regions
    ax.fill_between(rsi.index, 70, 100, alpha=0.1, color='#ff6b6b')
    ax.fill_between(rsi.index, 0, 30, alpha=0.1, color='#00ff88')

    _apply_dark_style(ax, f"{asset} RSI (3m timeframe)")
    ax.set_ylabel('RSI', color=COLORS['text'])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    plt.tight_layout()
    pil_img = _fig_to_pil(fig)
    plt.close(fig)

    return pil_img


def plot_funding_rate(data: Dict, asset: str) -> PILImage.Image:
    """Plot funding rate with positive/negative coloring."""
    funding = _get_series(data, 'funding_rate')
    if funding.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg'])

    # Color based on positive/negative
    colors = [COLORS['funding_pos'] if v >= 0 else COLORS['funding_neg'] for v in funding.values]

    ax.bar(funding.index, funding.values * 100, color=colors, width=0.02, alpha=0.8)
    ax.axhline(y=0, color=COLORS['text'], linestyle='-', alpha=0.5)

    _apply_dark_style(ax, f"{asset} Funding Rate")
    ax.set_ylabel('Funding Rate (%)', color=COLORS['text'])

    plt.tight_layout()
    pil_img = _fig_to_pil(fig)
    plt.close(fig)

    return pil_img


def plot_oi_vs_price(data: Dict, asset: str) -> PILImage.Image:
    """Plot Open Interest vs Price overlay."""
    price = _get_series(data, 'price')
    oi = _get_series(data, 'open_interest')

    if price.empty or oi.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    # Price on left axis
    ax1.plot(price.index, price.values, color=COLORS['price'], linewidth=1.5, label='Price')
    ax1.set_ylabel('Price (USDT)', color=COLORS['price'])
    ax1.tick_params(axis='y', labelcolor=COLORS['price'])

    # OI on right axis
    ax2 = ax1.twinx()
    ax2.plot(oi.index, oi.values, color=COLORS['oi'], linewidth=1.5, alpha=0.8, label='Open Interest')
    ax2.set_ylabel('Open Interest (USDT)', color=COLORS['oi'])
    ax2.tick_params(axis='y', labelcolor=COLORS['oi'])

    _apply_dark_style(ax1, f"{asset} Open Interest vs Price")
    ax2.spines['right'].set_color(COLORS['oi'])
    ax2.spines['right'].set_visible(True)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               facecolor=COLORS['panel'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

    plt.tight_layout()
    pil_img = _fig_to_pil(fig)
    plt.close(fig)

    return pil_img


def plot_borrow_repay(data: Dict, asset: str) -> PILImage.Image:
    """Plot Borrow/Repay volumes."""
    borrow = _get_series(data, 'total_borrow')
    repay = _get_series(data, 'total_repay')

    if borrow.empty and repay.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg'])

    if not borrow.empty:
        ax.plot(borrow.index, borrow.values, color=COLORS['borrow'], linewidth=1.5,
                label='Borrow', alpha=0.9)
    if not repay.empty:
        ax.plot(repay.index, repay.values, color=COLORS['repay'], linewidth=1.5,
                label='Repay', alpha=0.9)

    _apply_dark_style(ax, f"{asset} Borrow vs Repay (24H Rolling)")
    ax.set_ylabel('Volume (USDT)', color=COLORS['text'])
    ax.legend(loc='upper left', facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

    plt.tight_layout()
    pil_img = _fig_to_pil(fig)
    plt.close(fig)

    return pil_img


def plot_capital_flow(data: Dict, asset: str) -> PILImage.Image:
    """Plot net capital flow (borrow - repay)."""
    borrow = _get_series(data, 'total_borrow')
    repay = _get_series(data, 'total_repay')

    if borrow.empty or repay.empty:
        return None

    # Align and calculate net flow
    net_flow = borrow - repay
    net_flow = net_flow.dropna()

    if net_flow.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS['bg'])

    # Color bars based on positive/negative
    colors = [COLORS['borrow'] if v >= 0 else COLORS['repay'] for v in net_flow.values]

    ax.bar(net_flow.index, net_flow.values, color=colors, width=0.02, alpha=0.8)
    ax.axhline(y=0, color=COLORS['text'], linestyle='-', alpha=0.5)

    _apply_dark_style(ax, f"{asset} Net Capital Flow (Borrow - Repay)")
    ax.set_ylabel('Net Flow (USDT)', color=COLORS['text'])

    plt.tight_layout()
    pil_img = _fig_to_pil(fig)
    plt.close(fig)

    return pil_img


def generate_market_plots(data: Dict, asset: str) -> List[PILImage.Image]:
    """Generate all market plots and return as list of PIL Images."""
    plot_funcs = [
        (plot_price_action, "price_action"),
        (plot_rsi, "rsi"),
        (plot_funding_rate, "funding_rate"),
        (plot_oi_vs_price, "oi_vs_price"),
        (plot_borrow_repay, "borrow_repay"),
        (plot_capital_flow, "capital_flow"),
    ]

    plots = []
    for plot_func, name in plot_funcs:
        try:
            pil_image = plot_func(data, asset)
            if pil_image is not None:
                plots.append(pil_image)
        except Exception as e:
            logger.warning(f"Failed to generate {name} plot: {e}")

    return plots


def register_market_briefing(local_mcp_instance, csv_dir, requests_dir):
    """Register the market_briefing tool."""

    @local_mcp_instance.tool()
    def market_briefing(
        requester: str,
        asset: str,
        lookback_hours: int = 168,
        include_plots: bool = True
    ) -> list[Any]:
        """
        Get comprehensive market state briefing for an asset.

        Returns factual market data and visualizations WITHOUT trading
        instructions or predictions. Use for situational awareness.

        Parameters:
            requester (str): Identifier for logging (e.g., 'trading-agent', 'user-alex').
            asset (str): Asset symbol (BTC, ETH, SOL, etc.)
            lookback_hours (int): Hours of historical data (default: 168 = 7 days)
            include_plots (bool): Include chart images (default: True)

        Returns:
            List containing:
            - ImageContent: PNG charts (if include_plots=True)
            - TextContent: Markdown analysis sections

        Text Sections:
            - Price Action: Current price, changes, RSI, volatility, EMAs
            - Derivatives & Funding: Funding rate, crowding interpretation
            - Open Interest & Positioning: OI values and changes
            - Margin & Capital Flow: Borrow/repay volumes, net flow
            - Technical Indicators: RSI analysis with zone context

        Charts (6 total):
            - price_action: Price line with EMA 12/26
            - rsi: RSI with overbought/oversold zones
            - funding_rate: Funding rate histogram
            - oi_vs_price: Open Interest vs Price dual-axis
            - borrow_repay: Borrow and Repay volume lines
            - capital_flow: Net flow bar chart

        Example usage:
            market_briefing(requester="my-agent", asset="BTC")
            market_briefing(requester="my-agent", asset="ETH", include_plots=False)
            market_briefing(requester="my-agent", asset="SOL", lookback_hours=24)
        """
        logger.info(f"market_briefing invoked by {requester}: asset={asset.upper()}")

        try:
            asset_upper = asset.upper()

            # Fetch data
            data = _fetch_market_data(asset_upper, lookback_hours)

            # Generate text analysis
            analysis_text = generate_market_analysis(data, asset_upper)

            # Build response
            response: list[Any] = []

            # Add images first (if requested)
            if include_plots:
                pil_images = generate_market_plots(data, asset_upper)
                for pil_img in pil_images:
                    mcp_image = to_mcp_image(pil_img, format="png")
                    response.append(mcp_image.to_image_content())

            # Add text last
            response.append(analysis_text)

            # Log request
            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="market_briefing",
                input_params={
                    "asset": asset_upper,
                    "lookback_hours": lookback_hours,
                    "include_plots": include_plots
                },
                output_result=f"Generated briefing with {len(response) - 1} charts"
            )

            return response

        except ValueError as e:
            error_msg = f"Error generating briefing for {asset}: {str(e)}"
            logger.error(error_msg)

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="market_briefing",
                input_params={"asset": asset, "lookback_hours": lookback_hours},
                output_result=error_msg
            )

            return [error_msg]

        except Exception as e:
            error_msg = f"Unexpected error generating briefing for {asset}: {str(e)}"
            logger.error(error_msg)

            log_request(
                requests_dir=requests_dir,
                requester=requester,
                tool_name="market_briefing",
                input_params={"asset": asset, "lookback_hours": lookback_hours},
                output_result=error_msg
            )

            return [error_msg]
