"""Give journaled contracts readings from methods added after they were flagged.

The learning system grades each method on the contracts journaled since October 2026. A method
added later would start with no record, so its reading on each contract's flag date is rebuilt
from the daily prices up to that date. Idempotent: entries that already have the reading are left
alone, so it is safe to run on every research run.

Approximation: mean reversion's "market in balance" context comes from intraday volume profiles,
which were not kept for past days; backfilled readings use the daily trend context only.

    python -m options_dashboard.backfill          # backfill and save
    python -m options_dashboard.backfill --check  # report what would change, and mean reversion's edge
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any

import pandas as pd

from . import data, journal, structure, ta

METHOD = "meanrev"


def _reading(ind: pd.DataFrame, day: str) -> float | None:
    """Mean reversion as it read at the close before `day` (flags happen during that session)."""
    cut = pd.Timestamp(day, tz=ind.index.tz) if ind.index.tz is not None else pd.Timestamp(day)
    past = ind[ind.index < cut]
    if len(past) < 60:
        return None
    trend = ta.snapshot(past)["trend"]
    read = structure.mean_reversion(past, trend, None)
    return None if read is None else round(read["score"], 1)


def run(save: bool) -> dict[str, Any]:
    open_entries = journal.open_entries()
    outcome_files = sorted((journal.DIR / "outcomes").glob("*.jsonl"))
    outcomes = {p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
                for p in outcome_files}
    todo = [e for e in list(open_entries.values()) + [e for rows in outcomes.values() for e in rows]
            if e.get("methods") is not None and METHOD not in e["methods"]]
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for e in todo:
        by_ticker[e["ticker"]].append(e)
    filled = failed = 0
    for ticker, entries in by_ticker.items():
        try:
            ind = ta.add_indicators(data.history(ticker))
        except Exception:
            failed += len(entries)
            continue
        cache: dict[str, float | None] = {}
        for e in entries:
            day = e["entry_date"]
            if day not in cache:
                cache[day] = _reading(ind, day)
            if cache[day] is None:
                failed += 1
                continue
            e["methods"][METHOD] = cache[day]
            filled += 1
    if save and filled:
        journal._write(journal.OPEN_FILE, open_entries)
        for path, rows in outcomes.items():
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return {"needed": len(todo), "filled": filled, "failed": failed, "tickers": len(by_ticker)}


if __name__ == "__main__":
    check = "--check" in sys.argv
    result = run(save=not check)
    print(f"{'Would fill' if check else 'Filled'} {result['filled']} of {result['needed']} entries "
          f"across {result['tickers']} tickers; {result['failed']} without enough history")
