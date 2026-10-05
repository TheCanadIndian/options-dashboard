"""Streamlit dashboard. Run with: uv run streamlit run app.py"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from options_dashboard import alerts, config, paper, scanner, structure

# Chart colours (dark surface). Series hues are assigned in a fixed order.
SURFACE, GRID, AXIS = "#1a1a19", "#2c2c2a", "#383835"
INK, INK_MUTED = "#ffffff", "#898781"
BLUE, ORANGE, AQUA, RED = "#3987e5", "#d95926", "#199e70", "#e66767"

SCAN_KEYS = [
    "use_universe", "watchlist", "account_size", "risk_per_trade_pct", "min_dte", "max_dte", "min_delta",
    "max_delta", "min_open_interest", "min_volume", "max_spread_pct",
    "min_trend_strength", "fallback_risk_free_rate",
]

st.set_page_config(page_title="Options Dashboard", page_icon="📈", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def run_scan(scan_cfg_json: str) -> dict:
    # The JSON holds only the settings the sidebar can change (it doubles as the cache key);
    # everything else comes from the saved settings and defaults.
    return scanner.scan({**config.load(), **json.loads(scan_cfg_json)})


def sidebar(cfg: dict) -> dict:
    with st.sidebar.form("settings"):
        st.subheader("Account")
        cfg["account_size"] = st.number_input("Account size ($)", 100.0, 1e7, float(cfg["account_size"]), 100.0)
        cfg["risk_per_trade_pct"] = st.slider("Max premium per trade (% of account)", 1.0, 25.0, float(cfg["risk_per_trade_pct"]), 0.5)
        st.caption(f"Contracts costing more than ${config.max_premium(cfg):,.0f} are hidden.")

        st.subheader("What to scan")
        cfg["use_universe"] = st.toggle(
            "All liquid stocks and ETFs", bool(cfg["use_universe"]),
            help="Rebuilt each trading day from Yahoo's screener: US stocks with heavy volume whose options "
                 "have tight at-the-money spreads and real open interest, plus the main ETFs.",
        )
        cfg["watchlist"] = config.clean_watchlist(
            st.text_area("Always scan these tickers", " ".join(cfg["watchlist"]), height=100)
        )

        st.subheader("Contract filters")
        cfg["min_dte"], cfg["max_dte"] = st.slider("Days to expiry", 1, 180, (int(cfg["min_dte"]), int(cfg["max_dte"])))
        cfg["min_delta"], cfg["max_delta"] = st.slider("Delta range", 0.05, 0.95, (float(cfg["min_delta"]), float(cfg["max_delta"])), 0.05)
        cfg["min_open_interest"] = st.number_input("Min open interest", 0, 100000, int(cfg["min_open_interest"]), 50)
        cfg["min_volume"] = st.number_input("Min volume today", 0, 100000, int(cfg["min_volume"]), 5)
        cfg["max_spread_pct"] = st.slider("Max bid/ask spread (%)", 1.0, 50.0, float(cfg["max_spread_pct"]), 1.0)
        cfg["min_trend_strength"] = st.slider("Min trend strength", 0, 100, int(cfg["min_trend_strength"]), 5)

        st.subheader("Alerts")
        cfg["alert_min_score"] = st.slider("Alert when score is at least", 40, 100, int(cfg["alert_min_score"]))
        cfg["ntfy_topic"] = st.text_input("ntfy topic (phone push)", cfg["ntfy_topic"], help="Install the free ntfy app, subscribe to a hard-to-guess topic name, and enter it here.").strip()
        cfg["discord_webhook"] = st.text_input("Discord webhook URL", cfg["discord_webhook"], type="password").strip()

        if st.form_submit_button("Save settings", type="primary", width="stretch"):
            config.save(cfg)
            st.toast("Settings saved")

    if st.sidebar.button("Send test alert", width="stretch", disabled=not alerts.enabled(cfg)):
        errors = alerts.send(cfg, "Options Dashboard test", "Alerts are working.")
        st.sidebar.error("\n".join(errors)) if errors else st.sidebar.success("Test alert sent")
    return cfg


def contracts_table(df: pd.DataFrame, key: str) -> pd.Series | None:
    """Show contracts and return the row the user selected."""
    view = pd.DataFrame({
        "Score": df["score"],
        "Ticker": df["ticker"],
        "Type": df["type"].str.capitalize(),
        "Strike": df["strike"],
        "OTM": df["moneyness"] * 100,
        "Expiry": df["expiration"],
        "DTE": df["dte"],
        "Cost": df["cost"],
        "Delta": df["delta"],
        "Theta/day": df["theta"] * 100,
        "IV": df["iv"] * 100,
        "IV/HV": df["iv_hv"],
        "Breakeven move": df["breakeven_move"] * 100,
        "Chance of profit": df["pop"] * 100,
        "Spread": df["spread_pct"],
        "Open int.": df["openInterest"],
        "Earnings": df["earnings_before_expiry"].map({True: "Before expiry", False: ""}),
    })
    event = st.dataframe(
        view,
        key=key,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Strike": st.column_config.NumberColumn(format="$%g"),
            "OTM": st.column_config.NumberColumn(format="%+.1f%%", help="How far the stock must move to reach the strike. Negative means in the money."),
            "Cost": st.column_config.NumberColumn(format="$%.0f", help="Ask price x 100: what one contract costs and the most you can lose."),
            "Delta": st.column_config.NumberColumn(format="%.2f"),
            "Theta/day": st.column_config.NumberColumn(format="$%.2f", help="Dollars one contract loses per day from time decay."),
            "IV": st.column_config.NumberColumn(format="%.0f%%"),
            "IV/HV": st.column_config.NumberColumn(format="%.2f", help="Implied vol divided by 20-day realised vol. Below 1 means options are cheap relative to how the stock has been moving."),
            "Breakeven move": st.column_config.NumberColumn(format="%.1f%%", help="How far the stock must move by expiry for the contract to break even."),
            "Chance of profit": st.column_config.NumberColumn(format="%.0f%%", help="Model probability of finishing past breakeven at expiry."),
            "Spread": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    rows = event.selection.rows
    return df.iloc[rows[0]] if rows else None


def price_chart(ind: pd.DataFrame, ticker: str) -> go.Figure:
    d = ind.tail(180)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
        subplot_titles=(f"{ticker} price", "RSI (14)", "MACD histogram"),
    )
    for col, name, colour, width in (
        ("Close", "Close", INK, 2), ("ema20", "20 EMA", BLUE, 1.5),
        ("ema50", "50 EMA", ORANGE, 1.5), ("sma200", "200 SMA", AQUA, 1.5),
    ):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=name, line=dict(color=colour, width=width),
                                 hovertemplate="%{y:$,.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI", line=dict(color=BLUE, width=2),
                             showlegend=False, hovertemplate="%{y:.0f}"), row=2, col=1)
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=AXIS, width=1, dash="dot"), row=2, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["macd_hist"], name="MACD histogram", showlegend=False,
                         marker=dict(color=[BLUE if v >= 0 else RED for v in d["macd_hist"]], line_width=0),
                         hovertemplate="%{y:+.2f}"), row=3, col=1)
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK, family="system-ui, 'Segoe UI', sans-serif"),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#c3c2b7")), bargap=0.2,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=INK_MUTED),
                     rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=2, col=1)
    fig.update_annotations(font=dict(size=13, color="#c3c2b7"), x=0, xanchor="left")
    return fig


SURFACE_COLS = [("p10", "10Δ put"), ("p25", "25Δ put"), ("atm", "ATM"), ("c25", "25Δ call"), ("c10", "10Δ call")]


def _chart_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#c3c2b7"), x=0), height=300,
        margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK, family="system-ui, 'Segoe UI', sans-serif"),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(color="#c3c2b7")),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK_MUTED), tickformat=".0%")
    return fig


def smile_chart(surf: dict) -> go.Figure:
    """Implied volatility across deltas for the expiries nearest 2 weeks, 30 days and 60 days."""
    grid = pd.DataFrame(surf["grid"])
    labels = [label for _, label in SURFACE_COLS]
    fig = go.Figure()
    picked: list[int] = []
    for days, colour in ((14, BLUE), (30, ORANGE), (60, AQUA)):
        i = int((grid["dte"] - days).abs().idxmin())
        if i in picked:
            continue
        picked.append(i)
        row = grid.loc[i]
        fig.add_trace(go.Scatter(
            x=labels, y=[row[k] for k, _ in SURFACE_COLS], name=f"{row['expiration']} ({row['dte']}d)",
            mode="lines+markers", line=dict(color=colour, width=2), marker=dict(size=8),
            hovertemplate="%{x}: %{y:.1%}<extra>%{fullData.name}</extra>"))
    return _chart_layout(fig, "Volatility smile")


def surface_chart(surf: dict) -> go.Figure:
    """Implied volatility by expiry and delta; lighter is higher on the dark surface."""
    grid = pd.DataFrame(surf["grid"])
    z = grid[[k for k, _ in SURFACE_COLS]].to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=[label for _, label in SURFACE_COLS], y=[f"{e} ({d}d)" for e, d in zip(grid["expiration"], grid["dte"])],
        colorscale=[[0, "#184f95"], [0.5, "#3987e5"], [1, "#cde2fb"]], xgap=2, ygap=2,
        colorbar=dict(tickformat=".0%", tickfont=dict(color=INK_MUTED), thickness=10),
        hovertemplate="%{y}, %{x}: %{z:.1%}<extra></extra>"))
    fig = _chart_layout(fig, "Volatility surface")
    fig.update_yaxes(tickformat=None, autorange="reversed", gridcolor=SURFACE)
    return fig


def detail(ticker: str, info: dict, pick: pd.Series | None) -> None:
    st.subheader(f"{ticker}  ·  ${info['spot']:,.2f}  ·  {info['bias']} ({info['trend']:+.0f})")
    if pick is not None:
        st.markdown(f"**${pick['strike']:g} {pick['type']} expiring {pick['expiration']}** ({pick['dte']} days)")
        cols = st.columns(6)
        cols[0].metric("Cost / max loss", f"${pick['cost']:,.0f}")
        cols[1].metric("Breakeven", f"${pick['breakeven']:,.2f}", f"{pick['breakeven_move']:.1%} away", delta_color="off")
        cols[2].metric("Delta", f"{pick['delta']:+.2f}")
        cols[3].metric("Theta / day", f"-${abs(pick['theta']) * 100:.2f}", f"{pick['theta_pct']:.1%} of premium", delta_color="off")
        cols[4].metric("Vega", f"${pick['vega'] * 100:.2f}", "per 1 pt of IV", delta_color="off")
        cols[5].metric("Chance of profit", f"{pick['pop']:.0%}")
        if pick["earnings_before_expiry"]:
            st.warning(f"Earnings on {info.get('earnings')} fall before expiry. Implied volatility usually drops sharply afterwards, which can hurt a long option even if the stock moves your way.", icon="⚠️")
        if pick["stale"]:
            st.info("No live bid/ask (market closed or no quote). Prices are from the last trade.", icon="ℹ️")
    left, right = st.columns([3, 2])
    left.plotly_chart(price_chart(info["indicators"], ticker), width="stretch")
    if info.get("surface"):
        smile_col, surface_col = left.columns(2)
        smile_col.plotly_chart(smile_chart(info["surface"]), width="stretch")
        surface_col.plotly_chart(surface_chart(info["surface"]), width="stretch")
    left.caption(f"RSI {info['rsi']:.0f} · 20-day realised vol {info['hv20']:.0%} · daily range (ATR) {info['atr_pct']:.1%}")

    levels = info["levels"]
    named = [
        ("Call wall", levels.get("call_wall")), ("Value area high", levels.get("vah")),
        ("Point of control", levels.get("poc")), ("Value area low", levels.get("val")),
        ("Gamma flip", levels.get("flip")), ("Put wall", levels.get("put_wall")),
        ("Range high", levels.get("range_high")), ("Range low", levels.get("range_low")),
        ("Last price", info["spot"]),
    ]
    table = pd.DataFrame([(n, v) for n, v in named if v is not None], columns=["Level", "Price"])
    right.markdown("**Key levels**")
    right.dataframe(table.sort_values("Price", ascending=False), hide_index=True, width="stretch",
                    column_config={"Price": st.column_config.NumberColumn(format="$%.2f")})

    if "coverage" in levels:
        def shown(key: str) -> str:
            return "unknown" if levels.get(key) is None else structure.signed_short(levels[key])

        dex, gex, vanna, charm = right.columns(4)
        dex.metric("Customer DEX", shown("customer_dex"), delta_color="off",
                   help="Net delta customers hold through options, estimated from classified buy and sell flow. "
                        "Positive means net long (buying calls or selling puts).")
        gex.metric("Dealer gamma per 1%", shown("net_gex"),
                   None if levels.get("net_gex") is None else
                   ("Moves dampened" if levels["net_gex"] >= 0 else "Moves extended"), delta_color="off",
                   help="Dollar gamma dealers carry per 1% move, signed by the estimated customer positions.")
        vanna.metric("Vanna flow", shown("vanna_flow"), "per 1-pt fall in IV", delta_color="off",
                     help="Stock dealers buy (+) or sell (-) when implied volatility falls one point. The sign reverses when it rises.")
        charm.metric("Charm flow", shown("charm_flow"), "per day", delta_color="off",
                     help="Stock dealers buy (+) or sell (-) each day as time decay changes their delta.")
        right.caption(f"Classified flow explains {levels['coverage']:.0%} of open interest; the rest is treated "
                      "as unknown rather than as customer buys.")

    right.markdown("**Why this direction**")
    marks = {"+": "▲", "-": "▼", "!": "⚠️", "=": "•"}
    for method in info["methods"]:
        lines = "\n".join(f"- {marks[r[0]]} {r[2:]}" for r in method["reasons"])
        # A bare $ starts LaTeX in Streamlit markdown.
        right.markdown(f"{method['name']} ({method['score']:+.0f})\n{lines}".replace("$", "\\$"))


def paper_test(cfg: dict) -> None:
    df = paper.results()
    if df.empty:
        st.info("No paper trades yet. The background scanner logs the best pick per ticker during market hours.")
        return
    st.caption(
        f"Simulated trades: bought at the ask when first picked, valued at the latest bid. "
        f"First opened {df['opened'].min().replace('T', ' ')}, last priced {df['last_time'].max().replace('T', ' ')}."
    )
    tiles = st.columns(4)
    tiles[0].metric("Trades", len(df))
    tiles[1].metric("Winners", f"{int((df['pnl'] > 0).sum())} of {len(df)}")
    total = df["pnl"].sum()
    tiles[2].metric("Total P/L", f"{'-' if total < 0 else '+'}${abs(total):,.0f}",f"{df['pnl'].sum() / df['cost'].sum():+.1%} on ${df['cost'].sum():,.0f}", delta_color="off")
    tiles[3].metric("Median return", f"{df['return'].median():+.0%}")

    pct = st.column_config.NumberColumn(format="%+.0f%%")
    money = st.column_config.NumberColumn(format="$%.0f")
    summary = paper.summary(df, cfg["alert_min_score"])
    st.dataframe(
        pd.DataFrame({
            "Group": summary["group"], "Trades": summary["trades"].astype(int),
            "Winners": summary["winners"].astype(int), "Cost": summary["cost"], "P/L": summary["pnl"],
            "Return on cost": summary["return_on_cost"] * 100, "Median return": summary["median_return"] * 100,
        }),
        hide_index=True, width="stretch",
        column_config={"Cost": money, "P/L": st.column_config.NumberColumn(format="$%+.0f"),
                       "Return on cost": pct, "Median return": pct},
    )
    st.dataframe(
        pd.DataFrame({
            "Score": df["score"], "Ticker": df["ticker"], "Type": df["type"].str.capitalize(),
            "Strike": df["strike"], "Expiry": df["expiration"], "Opened": df["opened"].str.replace("T", " "),
            "Paid": df["cost"], "Now": df["value"], "P/L": df["pnl"], "Return": df["return"] * 100,
            "Best": df["best"] * 100, "Worst": df["worst"] * 100, "Stock move": df["stock_move"] * 100,
            "Status": df["status"].str.capitalize(),
        }),
        hide_index=True, width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Strike": st.column_config.NumberColumn(format="$%g"), "Paid": money, "Now": money,
            "P/L": st.column_config.NumberColumn(format="$%+.0f"), "Return": pct,
            "Best": st.column_config.NumberColumn(format="%+.0f%%", help="Highest return seen at any scan since entry."),
            "Worst": st.column_config.NumberColumn(format="%+.0f%%", help="Lowest return seen at any scan since entry."),
            "Stock move": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )


def main() -> None:
    cfg = sidebar(config.load())
    st.title("Options Dashboard")

    top = st.columns([1, 1, 4])
    if top[0].button("Rescan now", type="primary", width="stretch"):
        run_scan.clear()
    scan_key = json.dumps({k: cfg[k] for k in SCAN_KEYS}, sort_keys=True)
    scope = "the liquid universe" if cfg["use_universe"] else f"{len(cfg['watchlist'])} tickers"
    with st.spinner(f"Scanning {scope}. A full universe scan takes a few minutes..."):
        result = run_scan(scan_key)
    contracts, tickers = result["contracts"], result["tickers"]
    best = scanner.best_per_ticker(contracts)
    top[2].caption(
        f"Last scan {result['scanned_at']:%b %d, %H:%M} · Yahoo data, delayed about 15 minutes · "
        f"risk-free rate {result['rate']:.2%} · results cached for 15 minutes"
    )

    tiles = st.columns(4)
    tiles[0].metric("Max premium per trade", f"${config.max_premium(cfg):,.0f}")
    tiles[1].metric("Contracts passing filters", len(contracts))
    tiles[2].metric("Tickers with a setup", f"{len(best)} of {len(tickers)}")
    tiles[3].metric("Top score", f"{contracts['score'].max():.0f}" if len(contracts) else "None")
    if result["errors"]:
        st.warning("Could not scan: " + ", ".join(f"{t} ({e})" for t, e in result["errors"].items()))

    picks_tab, trends_tab, paper_tab, help_tab = st.tabs(
        ["Contracts", "Watchlist trends", "Paper test", "How scoring works"]
    )

    with paper_tab:
        paper_test(cfg)

    with picks_tab:
        if contracts.empty:
            st.info("No contracts passed the filters. Try a larger premium budget, a wider delta or expiry range, or cheaper tickers.")
        else:
            show_all = st.toggle("Show every passing contract, not just the best per ticker")
            shown = contracts if show_all else best
            st.caption("Select a row to see the chart and the reasoning.")
            pick = contracts_table(shown, "all" if show_all else "best")
            if pick is None:
                pick = shown.iloc[0]
            st.divider()
            detail(pick["ticker"], tickers[pick["ticker"]], pick)

    with trends_tab:
        short = {"auction": "Auction", "gamma": "Gamma", "wyckoff": "Wyckoff", "vpa": "VPA", "vol": "Vol",
                 "trend": "Trend"}
        trend_df = pd.DataFrame([
            {"Ticker": t, "Price": r["spot"], "Read": r["trend"], "Bias": r["bias"].capitalize(),
             **{short[m["key"]]: m["score"] for m in r["methods"]},
             "Wyckoff phase": r["levels"].get("phase", ""), "Contracts": len(r["contracts"])}
            for t, r in tickers.items()
        ]).sort_values("Read", ascending=False)
        signed = st.column_config.NumberColumn(format="%+.0f")
        st.dataframe(trend_df, hide_index=True, width="stretch", column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Read": st.column_config.NumberColumn(format="%+.0f", help="Combined score, -100 strongly bearish to +100 strongly bullish."),
            **{name: signed for name in short.values()},
        })
        choice = st.selectbox("Chart a ticker", list(tickers))
        if choice:
            detail(choice, tickers[choice], None)

    with help_tab:
        st.markdown(
            "Each ticker gets a **market read** from -100 to +100 that blends five methods. Bullish "
            "tickers are searched for calls, bearish ones for puts, and mixed ones are skipped."
        )
        st.dataframe(pd.DataFrame([
            ("Auction / market profile", structure.METHOD_WEIGHTS["auction"] * 100, "Price against the 10-day and prior-day value areas from 30-minute bars, and whether value is migrating."),
            ("Dealer gamma, vanna, charm and customer DEX", structure.METHOD_WEIGHTS["gamma"] * 100, "Which side holds each contract is estimated by classifying volume as bought at the ask or sold at the bid, every 15 minutes, over the last 10 sessions; unexplained open interest counts as unknown, not as buys. From that: customer delta exposure (net long or short delta), dealer gamma and its flip level, and the stock dealers must buy or sell as implied volatility (vanna) and time (charm) change their delta. Walls are where gamma is most concentrated. Readings are weighted by how much of the open interest the flow explains."),
            ("Wyckoff", structure.METHOD_WEIGHTS["wyckoff"] * 100, "Springs, upthrusts and breakouts of a 40-session range, or markup and markdown outside one."),
            ("Volume price analysis", structure.METHOD_WEIGHTS["vpa"] * 100, "Effort against result on the last five daily bars: no demand, no supply, stopping volume, climaxes."),
            ("Trend and momentum", structure.METHOD_WEIGHTS["trend"] * 100, "Moving averages, MACD, RSI and 20-day return."),
        ], columns=["Method", "Weight %", "What it reads"]), hide_index=True, width="stretch")
        st.markdown("Contracts that fit your budget and liquidity filters are then scored out of 100:")
        st.dataframe(pd.DataFrame([
            ("Market read", scanner.WEIGHTS["trend"], "How strongly the combined read agrees with the contract's direction."),
            ("Liquidity", scanner.WEIGHTS["liquidity"], "Tight bid/ask spread and healthy open interest, so you can get in and out."),
            ("Breakeven", scanner.WEIGHTS["breakeven"], "Move needed to break even compared with the move implied volatility expects."),
            ("IV value", scanner.WEIGHTS["iv_value"], "Implied vol vs recent realised vol. Cheaper options score higher."),
            ("Gamma backdrop", scanner.WEIGHTS["gamma"], "Negative dealer gamma (moves extend) and room to reach breakeven before the wall."),
            ("Theta", scanner.WEIGHTS["theta"], "Share of the premium lost to time decay each day."),
            ("Structural target", scanner.WEIGHTS["target"], "Breakeven inside the next level the stock is heading for (value area, point of control, gamma walls, flip, Wyckoff range), at least one ATR away. This is what lets an out-of-the-money strike score well."),
        ], columns=["Factor", "Points", "What it measures"]), hide_index=True, width="stretch")
        st.markdown(
            "Greeks are computed with Black-Scholes from the option's mid price, because Yahoo does "
            "not publish them. The profile spreads each 30-minute bar's volume evenly across its "
            "range, and gamma exposure assumes dealers are long calls and short puts, so both are "
            "estimates. A high score is a screen result, not a prediction: most long options "
            "expire worthless, and the chance-of-profit column is usually well under 50%."
        )

    st.caption("For education only. Not financial advice. Verify every quote with your broker before trading.")


main()
