"""Long intraday history from Alpaca's free market-data API, for research (Yahoo keeps 5- and
30-minute bars only 60 days). Regular-hours bars are cached in .history/ (git-ignored) and topped up.

Keys come from the environment (ALPACA_KEY_ID, ALPACA_SECRET_KEY) or the git-ignored .cloud.json.
A free Alpaca account (paper trading, no money needed) is enough; its historical bars reach back to
2016, with the last 15 minutes withheld.

    python -m options_dashboard.history keys            # asks for the keys and saves them, hidden
    python -m options_dashboard.history fetch SPY 5Min  # download or top up the cache
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import HOME

URL = "https://data.alpaca.markets/v2/stocks/bars"
CLOUD = HOME / ".cloud.json"
CACHE = HOME / ".history"  # outside .state/, which is synced with the cloud as one archive
START = "2016-01-01"


def _keys() -> tuple[str, str]:
    key, secret = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET_KEY")
    if not (key and secret):
        try:
            saved = json.loads(CLOUD.read_text(encoding="utf-8"))
            key, secret = saved["alpaca_key_id"], saved["alpaca_secret_key"]
        except (OSError, KeyError, json.JSONDecodeError):
            sys.exit("No Alpaca keys: run  python -m options_dashboard.history keys")
    return key, secret


def save_keys() -> None:
    """Ask for the keys without echoing them and add them to .cloud.json."""
    try:
        saved = json.loads(CLOUD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        saved = {}
    saved["alpaca_key_id"] = getpass.getpass("Alpaca API key ID (hidden): ").strip()
    saved["alpaca_secret_key"] = getpass.getpass("Alpaca secret key (hidden): ").strip()
    CLOUD.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    print(f"Saved to {CLOUD} (git-ignored). Checking them...")
    frame = _download("SPY", "1Day", datetime.now(timezone.utc) - timedelta(days=10), None)
    print(f"Keys work: {len(frame)} recent SPY daily bars received." if len(frame) else "No bars came back.")


def _download(symbol: str, timeframe: str, start: datetime, end: datetime | None) -> pd.DataFrame:
    key, secret = _keys()
    # The free plan serves the full consolidated tape (SIP) except the last 15 minutes.
    end = end or datetime.now(timezone.utc) - timedelta(minutes=20)
    params = {"symbols": symbol, "timeframe": timeframe, "start": start.isoformat(), "end": end.isoformat(),
              "limit": 10000, "adjustment": "raw", "feed": "sip"}
    rows, session = [], requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    while True:
        r = session.get(URL, params=params, timeout=60)
        if r.status_code == 403 and params["feed"] == "sip":  # plans without the tape: the IEX feed
            params["feed"] = "iex"
            continue
        r.raise_for_status()
        body = r.json()
        rows += body.get("bars", {}).get(symbol, [])
        if not body.get("next_page_token"):
            break
        params["page_token"] = body["next_page_token"]
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = pd.DataFrame(rows).rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    frame.index = pd.to_datetime(frame.pop("t"), utc=True).dt.tz_convert("America/New_York")
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def bars(symbol: str = "SPY", timeframe: str = "5Min", refresh: bool = True) -> pd.DataFrame:
    """Regular-hours (9:30-16:00 New York) bars since 2016, cached and topped up on request."""
    path = CACHE / f"{symbol}_{timeframe}.csv.gz"
    cached = None
    if path.exists():
        cached = pd.read_csv(path, index_col=0)
        cached.index = pd.to_datetime(cached.index, utc=True).tz_convert("America/New_York")
    if cached is not None and not refresh:
        return cached
    start = (cached.index[-1].tz_convert("UTC").to_pydatetime() + timedelta(minutes=1) if cached is not None and len(cached)
             else datetime.fromisoformat(START).replace(tzinfo=timezone.utc))
    new = _download(symbol, timeframe, start, None)
    if not timeframe.endswith("Day") and len(new):
        minutes = new.index.hour * 60 + new.index.minute
        new = new[(minutes >= 570) & (minutes < 960)]
    out = pd.concat([cached, new]) if cached is not None else new
    out = out[~out.index.duplicated(keep="last")].sort_index()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, compression="gzip")
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["keys"]:
        save_keys()
    elif sys.argv[1:2] == ["fetch"]:
        args = sys.argv[2:]
        sym, tf = (args[0] if args else "SPY"), (args[1] if len(args) > 1 else "5Min")
        b = bars(sym, tf)
        print(f"{sym} {tf}: {len(b):,} regular-hours bars, {b.index[0]} to {b.index[-1]}")
    else:
        print(__doc__)
