"""Move the scanner's state to and from the Cloudflare Worker that stores it in R2.

The scan runs on GitHub Actions, which starts with an empty disk each time. Before a scan the
state (caches, journal, simulated accounts) is pulled as one archive; afterwards it is pushed
back. A push is refused unless this machine pulled successfully first, so a run that could not
load the state can never overwrite it with a fresh, empty one.

Settings come from the environment (STATE_URL, STATE_TOKEN) or, on the PC, from .cloud.json.

    python -m options_dashboard.cloudstate pull      # before a scan, or to refresh the PC's copy
    python -m options_dashboard.cloudstate push      # after a scan
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
from pathlib import Path

import requests

from .config import HOME

DIRS = (".state", "journal", "paper")
MARKER = HOME / ".state" / ".pulled"
REQUIRED = HOME / "paper" / "portfolio.json"  # present in any real state


def _settings() -> tuple[str, str]:
    url, token = os.environ.get("STATE_URL"), os.environ.get("STATE_TOKEN")
    if not (url and token):
        try:
            saved = json.loads((HOME / ".cloud.json").read_text(encoding="utf-8"))
            url, token = saved["state_url"], saved["state_token"]
        except (OSError, KeyError, json.JSONDecodeError):
            sys.exit("Set STATE_URL and STATE_TOKEN, or create .cloud.json with state_url and state_token.")
    return url.rstrip("/") + "/state", token


def pull() -> None:
    url, token = _settings()
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        names = tar.getnames()
        if not all(n.split("/")[0] in DIRS for n in names):
            sys.exit("Refusing an archive with files outside the state folders.")
        tar.extractall(HOME, filter="data")
    if not REQUIRED.exists():
        sys.exit(f"Pulled state has no {REQUIRED.relative_to(HOME)}; refusing to continue.")
    MARKER.write_text(r.headers.get("Last-Modified", ""), encoding="utf-8")
    print(f"Pulled {len(names)} files ({len(r.content) / 1e6:.1f} MB), saved {r.headers.get('Last-Modified')}")


def push() -> None:
    if not MARKER.exists():
        sys.exit("Not pushing: this machine has not pulled the state in this run.")
    if not REQUIRED.exists():
        sys.exit(f"Not pushing: {REQUIRED.relative_to(HOME)} is missing.")
    url, token = _settings()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in DIRS:
            if (HOME / d).exists():
                tar.add(HOME / d, arcname=d, filter=lambda t: None if t.name.endswith(".pulled") else t)
    r = requests.put(url, data=buf.getvalue(), headers={"Authorization": f"Bearer {token}"}, timeout=300)
    r.raise_for_status()
    print(f"Pushed {buf.tell() / 1e6:.1f} MB")


def seed() -> None:
    """First upload from the PC, before the cloud has any state."""
    MARKER.parent.mkdir(exist_ok=True)
    MARKER.write_text("seeded from this machine", encoding="utf-8")
    push()


if __name__ == "__main__":
    {"pull": pull, "push": push, "seed": seed}[sys.argv[1]]()
