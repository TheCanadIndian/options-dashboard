"""Command-line scanner: run once, or loop during market hours and send alerts."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from . import alerts, config, data, journal, paper, portfolio, scanner, scorecard, site

market_open = data.market_open


def run_once(send_alerts: bool, log_paper: bool) -> None:
    cfg = config.load()
    result = scanner.scan(cfg)
    top = scanner.best_per_ticker(result["contracts"]).head(15)
    print(f"\n{result['scanned_at']:%Y-%m-%d %H:%M}  scanned {len(result['tickers'])} tickers, "
          f"{len(result['contracts'])} contracts passed filters "
          f"(portfolio buys up to ${config.max_premium(cfg):.0f})")
    if top.empty:
        print("No contracts passed the filters.")
    for _, r in top.iterrows():
        print(
            f"  {r['score']:5.1f}  {r['ticker']:<5} {r['type']:<4} ${r['strike']:<7g} {r['expiration']}"
            f"  cost ${r['cost']:<5.0f} delta {r['delta']:+.2f}  IV {r['iv']:.0%}"
            f"  POP {r['pop']:.0%}{'  (stale quote)' if r['stale'] else ''}"
        )
    for ticker, err in result["errors"].items():
        print(f"  ! {ticker}: {err}")
    if log_paper:
        opened, marked = paper.update(result, cfg)
        print(f"Paper trades: {opened} opened, {marked} re-priced")
        states = portfolio.run_all(result, cfg)
        state = states["core"]
        ranked = sorted(states.items(), key=lambda kv: -portfolio.equity(kv[1]))
        print("Accounts: " + ", ".join(f"{k} ${portfolio.equity(v):,.0f}" for k, v in ranked))
        print(f"Portfolio: ${portfolio.equity(state):,.2f} "
              f"({len(state['positions'])} open, {len(state['closed'])} closed, cash ${state['cash']:,.2f})")
        logged = journal.record_scan(result, cfg)
        print(f"Journal: {logged['added']} added, {logged['updated']} followed, {logged['matured']} graded")
        if scorecard.due():
            card = scorecard.build(cfg, scanner.WEIGHTS, scanner.FULL_CONVICTION)
            print(f"Scorecard rebuilt: {card['tuning']['message']}")
        site.build(state, cfg, result)
        if cfg["site_push"]:
            error = site.publish()
            if error:
                print(f"  ! site not published: {error}")
    if send_alerts:
        sent, errors = alerts.notify(result, cfg)
        if alerts.enabled(cfg):
            print(f"Alerts sent: {sent}")
        for err in errors:
            print(f"  ! alert failed: {err}")


def report() -> None:
    cfg = config.load()
    df = paper.results()
    if df.empty:
        print("No paper trades logged yet.")
        return
    print(f"Paper test: {len(df)} trades, first opened {df['opened'].min()}, "
          f"last priced {df['last_time'].max()}\n")
    for _, r in df.iterrows():
        print(
            f"  {r['score']:5.1f}  {r['ticker']:<5} {r['type']:<4} ${r['strike']:<7g} {r['expiration']}"
            f"  paid ${r['cost']:<5.0f} now ${r['value']:<5.0f} P/L ${r['pnl']:+7.0f} ({r['return']:+.0%})"
            f"  best {r['best']:+.0%} worst {r['worst']:+.0%}  stock {r['stock_move']:+.1%}"
        )
    print()
    for _, s in paper.summary(df, cfg["alert_min_score"]).iterrows():
        print(
            f"  {s['group']:<18} {int(s['trades']):>2} trades, {int(s['winners'])} winners, "
            f"cost ${s['cost']:,.0f}, P/L ${s['pnl']:+,.0f} ({s['return_on_cost']:+.1%}), "
            f"median {s['median_return']:+.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan the watchlist for options contracts.")
    parser.add_argument("--loop", type=int, metavar="MINUTES", help="rescan on this interval")
    parser.add_argument("--no-alert", action="store_true", help="do not send alerts")
    parser.add_argument("--no-paper", action="store_true", help="do not log paper trades")
    parser.add_argument("--market-hours", action="store_true", help="do nothing when the US market is closed")
    parser.add_argument("--report", action="store_true", help="show paper-trade results and exit")
    args = parser.parse_args()

    if sys.stdout is None:  # started without a console (scheduled task): keep a log instead
        config.STATE_DIR.mkdir(exist_ok=True)
        sys.stdout = sys.stderr = open(config.STATE_DIR / "scan.log", "a", encoding="utf-8", buffering=1)

    if args.report:
        report()
        return
    if not args.loop:
        if not args.market_hours or market_open():
            run_once(not args.no_alert, not args.no_paper)
        return
    print(f"Scanning every {args.loop} minutes. Ctrl+C to stop.")
    while True:
        if not args.market_hours or market_open():
            try:
                run_once(not args.no_alert, not args.no_paper)
            except Exception as exc:
                print(f"Scan failed: {exc}")
        else:
            print(f"{datetime.now():%H:%M}  market closed, waiting")
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    main()
