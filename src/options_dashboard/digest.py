"""Daily Discord digest, posted once each market day right after the 10:00 New York scan.

Four posts (Discord caps a message at 6,000 characters):
  1. Market: SPY and the regime, with pictures of the sector and Mag 7 rotation maps
  2. Earnings calendar: who reports in the next week, with EPS and revenue expectations and the
     move the options market prices in
  3. Contracts scoring 70 or more on the 10:00 scan, swing and short-dated
  4. The strategy race (trades since the last digest) and the Roth research (buy-area entries
     and exits since the last digest)

10:00 is the first scan with live quotes: before 9:45 the 15-minute-delayed data still shows the
pre-open book. Each scan saves what this needs (save_scan) to .state/last_scan.json. The maps are
the site's own drawings, screenshotted with a headless Chrome or Edge. The webhook comes from the
DISCORD_WEBHOOK environment variable (a GitHub secret in the cloud) or "discord_digest_webhook" in
config.json, separate from the per-scan contract alerts.

    python -m options_dashboard.digest            # build and post
    python -m options_dashboard.digest --daily    # post only once a market day, from 10:00 on
    python -m options_dashboard.digest --dry-run  # print the text and save the images instead
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, time as clock, timedelta
from html import escape
from pathlib import Path
from typing import Any

import requests

from . import config, data, longterm, regime, sectors, site
from .config import STATE_DIR

SITE = "https://thecanadindian.github.io/options-dashboard/"
SEEN = STATE_DIR / "digest.json"
LAST_SCAN = STATE_DIR / "last_scan.json"
MIN_SCORE = 70
EARNINGS_DAYS = 7
AMBER, GREEN, RED, BLUE = 0xFFA028, 0x2FD36B, 0xFF5252, 0x3987E5
MARKS = {"leading": "🟢 Leading", "improving": "🔵 Improving", "weakening": "🟠 Weakening", "lagging": "🔴 Lagging"}


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:+.{digits}%}"


def _money(x: float | None) -> str:
    if x is None:
        return "n/a"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= size:
            return f"${x / size:,.2f}{unit}"
    return f"${x:,.2f}"


# ---------------------------------------------------------------- saved by every scan

def save_scan(result: dict[str, Any]) -> None:
    """Keep the last scan's high scorers and earnings outlooks for the digest."""
    def rows(frame) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        keep = frame[(frame["score"] >= MIN_SCORE) & ~frame["stale"]].sort_values("score", ascending=False)
        return [{"ticker": r["ticker"], "type": r["type"], "strike": float(r["strike"]), "expiration": r["expiration"],
                 "dte": int(r["dte"]), "score": float(r["score"]), "ask": float(r["ask"]), "bid": float(r["bid"]),
                 "delta": float(r["delta"]), "iv": float(r["iv"]), "pop": float(r["pop"]),
                 "breakeven_move": float(r["breakeven_move"]), "spread_pct": float(r["spread_pct"]),
                 "earnings": bool(r["earnings_before_expiry"])} for _, r in keep.iterrows()]

    earnings = []
    for t, info in result["tickers"].items():
        o = info.get("earnings_outlook")
        if o:
            earnings.append({k: o.get(k) for k in ("ticker", "date", "timing", "eps_estimate", "eps_year_ago",
                                                    "revenue_estimate", "implied_move", "historical_move", "lean",
                                                    "beat_probability", "analysts")})
    spy = (result["tickers"].get("SPY") or {}).get("levels") or {}
    out = {"time": result["scanned_at"].isoformat(timespec="minutes"), "swing": rows(result["contracts"]),
           "spy_gamma": {k: spy.get(k) for k in ("net_gex", "flip", "call_wall", "put_wall")},
           "short": rows(result.get("short_contracts")), "earnings": earnings}
    LAST_SCAN.parent.mkdir(parents=True, exist_ok=True)
    LAST_SCAN.write_text(json.dumps(out, default=float), encoding="utf-8")


def _last_scan() -> dict[str, Any]:
    try:
        return json.loads(LAST_SCAN.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _seen() -> dict[str, Any]:
    try:
        return json.loads(SEEN.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------- 1. market and rotation pictures

def _browser() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for path in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(path).exists():
            return path
    return None


def _rotation_png(rot: dict[str, Any], title: str, stocks: bool) -> bytes | None:
    """The site's rotation map for these series, as a PNG (None without a browser)."""
    browser = _browser()
    if browser is None or not rot:
        return None
    weeks = site.MAG7_TAIL_WEEKS + 1 if stocks else None
    tails = {n: (v["tail"][-weeks:] if weeks else v["tail"]) for n, v in rot.items() if v.get("tail")}
    palette = ({n: f"--s{i}" for i, n in enumerate(sorted(tails, key=lambda n: rot[n]["etf"]), 1)} if stocks else None)
    svg = site._rotation_svg(rot, tails, (760, 560, 64, 24, 24, 56), "x", title, palette)
    if palette:
        legend = "".join(f'<span><i style="background:var({palette[n]})"></i>{escape(rot[n]["etf"])}</span>' for n in palette)
    else:
        legend = "".join(f'<span><i style="background:var({site.QUADRANTS[q][0]})"></i>{q.capitalize()}</span>'
                         for q in ("leading", "weakening", "lagging", "improving"))
    page = (f'<!doctype html><html><head><meta charset="utf-8"><style>{site.CSS}'
            f'body{{margin:0;padding:12px 16px;width:780px}}</style></head><body>'
            f'<h2 style="margin:0 0 8px">{escape(title)} · {datetime.now(data.NEW_YORK):%b %d, %Y}</h2>'
            f'<div class="legend">{legend}</div>{svg}</body></html>')
    tmp = Path(tempfile.mkdtemp())
    html, png = tmp / "map.html", tmp / "map.png"
    html.write_text(page, encoding="utf-8")
    try:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                        "--force-device-scale-factor=2", f"--screenshot={png}", "--window-size=812,650",
                        html.as_uri()], capture_output=True, timeout=90)
        # On Windows the launcher can exit before the browser has written the file.
        for _ in range(40):
            if png.exists() and png.stat().st_size > 0:
                time.sleep(0.3)
                return png.read_bytes()
            time.sleep(0.5)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _quadrant_changes(rot: dict[str, Any], before: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    now = {r["etf"]: r["quadrant"] for r in rot.values()}
    moved = [f"{t} {before[t]} → {q}" for t, q in now.items() if before.get(t) and before[t] != q]
    return moved, now


def _market(seen: dict[str, Any], keep: dict[str, Any], files: dict[str, bytes]) -> list[dict[str, Any]]:
    """SPY and the regime as text, then the two rotation maps as pictures."""
    lines, colour = [], AMBER
    try:
        spy = data.history("SPY")["Close"]
        lines.append(f"**SPY** {spy.iloc[-1]:,.2f} ({_pct(spy.iloc[-1] / spy.iloc[-2] - 1)} on the day)")
    except Exception:
        pass
    try:
        m = regime.read(*regime.last_breadth())
        colour = GREEN if m["state"].startswith("risk-on") else RED
        lines.append(f"**Regime:** {m['state']} ({m['score']:+.0f}, {m['change']:+.0f} over 10 sessions) · "
                     f"VIX {m['vix'][0]:.2f} ({_pct(m['vix'][0] / m['vix'][1] - 1)})")
        odds = m["odds"].get("drop_odds")
        if odds is not None:
            lines.append(f"Odds of a 5%+ SPY drop within a month: **{odds:.0%}** (usually {m['base_drop']:.0%})")
        lines += [f"⚠ {w}" for w in m["warnings"][:2]]
    except Exception as exc:
        lines.append(f"Regime unavailable ({type(exc).__name__})")
    embeds = [{"title": "Market", "description": "\n".join(lines)[:4000], "color": colour}]
    for title, symbols, key in (("Sector rotation", None, "sectors"), ("Mag 7 rotation", sectors.MAG7, "mag7")):
        try:
            rot = sectors.rotation(symbols)
        except Exception as exc:
            embeds.append({"title": title, "description": f"Unavailable ({type(exc).__name__})", "color": AMBER})
            continue
        moved, keep[key] = _quadrant_changes(rot, seen.get(key, {}))
        embed = {"title": title, "color": AMBER,
                 "description": ("Changed quadrant: " + ", ".join(moved)) if moved else "No quadrant changes since the last digest."}
        png = _rotation_png(rot, title, symbols is not None)
        if png:
            name = f"{key}-rotation.png"
            files[name] = png
            embed["image"] = {"url": f"attachment://{name}"}
        else:  # no browser to draw with: fall back to a text list
            embed["description"] += "\n" + "\n".join(
                f"{MARKS[q]}: " + ", ".join(r["etf"] for r in rot.values() if r["quadrant"] == q)
                for q in ("leading", "improving", "weakening", "lagging"))
        embeds.append(embed)
    return embeds


# ---------------------------------------------------------------- 2. earnings calendar

def _earnings(scan: dict[str, Any]) -> dict[str, Any]:
    today = date.today()
    soon = [e for e in scan.get("earnings", [])
            if e.get("date") and 0 <= (date.fromisoformat(e["date"]) - today).days <= EARNINGS_DAYS]
    soon.sort(key=lambda e: (e["date"], e["ticker"]))
    if not soon:
        text = f"No reports in the next {EARNINGS_DAYS} days among the scanned tickers."
    else:
        lines = []
        for e in soon[:20]:
            d = date.fromisoformat(e["date"])
            when = f"{d:%a %b} {d.day}, {'before open' if e.get('timing') == 'before open' else 'after close'}"
            eps = (f"EPS est {e['eps_estimate']:.2f}" + (f" (yr ago {e['eps_year_ago']:.2f})" if e.get("eps_year_ago") else "")
                   if e.get("eps_estimate") is not None else "EPS est n/a")
            move = (f"options price ±{e['implied_move']:.1%}" + (f" (usually ±{e['historical_move']:.1%})"
                    if e.get("historical_move") else "") if e.get("implied_move") else "implied move n/a")
            beat = f" · beat odds {e['beat_probability']:.0%}" if e.get("beat_probability") is not None else ""
            lines.append(f"**{e['ticker']}** · {when}\n{eps} · revenue est {_money(e.get('revenue_estimate'))} · "
                         f"{move} · lean {e.get('lean') or 'n/a'}{beat}")
        text = "\n".join(lines)
        if len(soon) > 20:
            text += f"\n…and {len(soon) - 20} more on the Earnings page"
    return {"title": f"Earnings calendar: next {EARNINGS_DAYS} days", "description": text[:4000], "color": BLUE}


# ---------------------------------------------------------------- 3. contracts scoring 70+

def _contracts(scan: dict[str, Any]) -> dict[str, Any]:
    """One contract per ticker: the highest-scoring one, which is always in the direction the model
    expects (calls when its read is bullish, puts when bearish)."""
    best: dict[str, dict[str, Any]] = {}
    for r in scan.get("swing", []):  # already sorted best first
        best.setdefault(r["ticker"], r)
    lines = []
    for r in sorted(best.values(), key=lambda r: -r["score"])[:15]:
        exp = date.fromisoformat(r["expiration"])
        way = "▲ Bullish → buy call" if r["type"] == "call" else "▼ Bearish → buy put"
        warn = " · ⚠ earnings before expiry" if r["earnings"] else ""
        lines.append(f"**{r['ticker']}** {way}: ${r['strike']:g} {exp:%b} {exp.day} ({r['dte']}d) · ask ${r['ask']:.2f} "
                     f"(${r['ask'] * 100:,.0f}) · score {r['score']:.0f} · needs {r['breakeven_move']:.1%}{warn}")
    text = "\n".join(lines) or f"No contract scored {MIN_SCORE} or more on the last scan."
    if scan.get("time"):
        text += (f"\n\nBest contract per ticker from the {scan['time'][11:16]} ET scan, scored {MIN_SCORE}+. "
                 "Buy to open only; prices are 15 minutes delayed.")
    return {"title": "Today's contract per ticker", "description": text[:4000], "color": AMBER}


def _zerodte() -> dict[str, Any]:
    """The 0DTE lab's forward test: results per variant and the last trading day's trades."""
    from . import zerodte
    state = zerodte.load()
    lines = []
    for s in zerodte.summary(state):
        avg = "n/a" if s["avg"] is None else f"${s['avg']:+.0f}"
        lines.append(f"**{s['name']}**: {s['trades']} trades, {s['wins']} won, avg {avg}, total ${s['total']:+,.0f}")
    days = sorted(state["days"])
    if days:
        last = days[-1]
        trades = [t for t in state["trades"] if t["date"] == last]
        lines.append(f"\n**{last}:** " + ("; ".join(
            f"{t['variant']} {t['ticker']} {t['strike']:g} {t['type']} at ${t['entry']:.2f} → "
            + (f"{'+' if t['pnl'] >= 0 else '-'}${abs(t['pnl']):.0f}" if t["status"] == "closed" else "open")
            for t in trades) or "no signal"))
    lines.append("Forward test at real prices of the 3:30 PM momentum rule; paper only.")
    return {"title": "0DTE lab (end-of-day momentum)", "description": "\n".join(lines)[:4000], "color": AMBER}


# ---------------------------------------------------------------- 4. Roth research

def _invest(seen: dict[str, Any], keep: dict[str, Any]) -> dict[str, Any] | None:
    res = longterm.load()
    if not res:
        return None
    picks = res["shortlist"]
    now = {r["ticker"]: r["buy_area"]["status"] for r in picks}
    keep["status"] = now
    before = seen.get("status", {})
    buyable = ("In buy area", "Near buy area")
    entered = [t for t, s in now.items() if s in buyable and before.get(t) not in buyable]
    left = [t for t, s in now.items() if s not in buyable and before.get(t) in buyable]
    lines = []
    if before:
        lines.append(f"**Moved into a buy area:** {', '.join(entered)}" if entered else "No new buy-area entries since the last digest.")
        if left:
            lines.append(f"**Moved out:** {', '.join(left)}")
    lines.append("**In or near a buy area now:**")
    in_area = [r for r in picks if r["buy_area"]["status"] in buyable and r["buy_area"].get("valuation") != "expensive even here"]
    for r in sorted(in_area, key=lambda r: -r["overall"])[:8]:
        a = r["buy_area"]
        lines.append(f"• **{r['ticker']}** ${r['price']:,.2f} · area ${a['low']:,.2f}–{a['high']:,.2f} · base target "
                     f"{_pct(r['base'] / r['price'] - 1, 0)}{' ⚠ de-rated' if r.get('derated') else ''}")
    if res.get("themes"):
        lines.append("**Themes:** " + " · ".join(f"{t['name']}: {t['verdict'].split(':')[0]}" for t in res["themes"]))
    return {"title": "Roth research (S&P 500 shortlist)", "description": "\n".join(lines)[:4000], "color": BLUE}


# ---------------------------------------------------------------- assemble and post

def build(cfg: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], dict[str, bytes]]], dict[str, Any]]:
    """The messages to post (each with its image attachments), and what to remember for next time."""
    seen, keep, scan = _seen(), {}, _last_scan()
    day = datetime.now(data.NEW_YORK)
    footer = {"description": f"[Open the dashboard]({SITE}) · Research, not financial advice.", "color": AMBER}
    files: dict[str, bytes] = {}
    market = _market(seen, keep, files)
    invest = _invest(seen, keep)
    messages = [
        ({"content": f"**📊 Daily digest · {day:%A, %B} {day.day}**", "embeds": market}, files),
        ({"embeds": [_earnings(scan)]}, {}),
        ({"embeds": [_contracts(scan), _zerodte()]}, {}),
        ({"embeds": ([invest] if invest else []) + [footer]}, {}),
    ]
    for m, _ in messages:
        m["username"] = "Options Terminal"
    return messages, keep


def post(messages: list[tuple[dict[str, Any], dict[str, bytes]]], url: str) -> None:
    for payload, files in messages:
        if files:
            upload = {f"files[{i}]": (name, png, "image/png") for i, (name, png) in enumerate(files.items())}
            r = requests.post(url, data={"payload_json": json.dumps(payload)}, files=upload, timeout=30)
        else:
            r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        time.sleep(1)  # Discord rate-limits webhooks; a short pause keeps the posts in order


DAILY_AT = clock(10, 0)  # New York: the first scan with live quotes


def due() -> bool:
    """Once a market day, from DAILY_AT on."""
    now = datetime.now(data.NEW_YORK)
    return (data.market_open() and now.time() >= DAILY_AT
            and _seen().get("daily_date", (_seen().get("date") or "")[:10]) != datetime.now().date().isoformat())


if __name__ == "__main__":
    if "--daily" in sys.argv and not due():
        print("Digest not due")
        raise SystemExit
    cfg = config.load()
    url = os.environ.get("DISCORD_WEBHOOK") or cfg.get("discord_digest_webhook")
    if "--daily" in sys.argv and not url:
        print("Digest skipped: no DISCORD_WEBHOOK set")
        raise SystemExit
    messages, keep = build(cfg)
    if "--dry-run" in sys.argv or not url:
        for payload, files in messages:
            print(json.dumps(payload, indent=1, ensure_ascii=False))
            for name, png in files.items():
                (STATE_DIR / name).write_bytes(png)
                print(f"[image saved to {STATE_DIR / name}]")
        if not url:
            print("No webhook set (DISCORD_WEBHOOK), so nothing was posted.", file=sys.stderr)
    else:
        post(messages, url)
        SEEN.parent.mkdir(parents=True, exist_ok=True)
        stamp = {"date": datetime.now().isoformat(timespec="minutes"), "daily_date": _seen().get("daily_date")}
        if "--daily" in sys.argv:  # only the scheduled post marks the day done; a test post does not
            stamp["daily_date"] = datetime.now().date().isoformat()
        SEEN.write_text(json.dumps(keep | stamp), encoding="utf-8")
        print(f"Digest posted to Discord ({len(messages)} messages)")
