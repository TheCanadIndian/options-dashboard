# Options Dashboard

A free options scanner and research site for one person's small accounts. It finds long call
and put contracts using dealer positioning and technical structure, runs simulated (paper)
accounts on them, grades every flagged contract on dollar profit to tune itself, and keeps a
long-term research page for a real-money Roth IRA. The site is public:
https://thecanadindian.github.io/options-dashboard/ (built into `docs/`).

This repo is public. Never commit keys, account numbers, real holdings or dollar amounts of
the user's real money.

## The user's rules (follow these without being reminded)

- **Buy to open only.** Long calls and long puts, closed by selling. No spreads, no writing or
  selling options to open, in any account, scan or suggestion.
- **Tight stops, near-ideal entries.** When trades get stopped out, improve the entry (closeness
  to a level that proves the trade wrong, reward against risk), never widen the stop.
- **Methods they trust:** auction theory and market profile, volume price analysis, Wyckoff,
  dealer gamma (plus vanna and charm, with customer flow deciding which side holds open
  interest), skew and volatility surfaces, sector rotation, seasonality, AVWAPs anchored at
  quarters and volatility events, regime changes, and mean reversion (added Oct 2026: it fades
  stretches in balance and buys pullbacks in trends, but never fades a strong trend's own direction).
- **Grade on profit**, in dollars per contract, not on statistical fit.
- **Swings first.** Weeklies and same-day gamma trades are welcome as separate experiments.
- **Roth IRA (Schwab, started with $500, adds money occasionally): research only.** The Invest
  page never shows their holdings or dollar amounts. Frame it as research, not advice. Schwab's
  fractional shares (Stock Slices) cover only the S&P 500; anything else is whole shares.
- They like plain-language explanations ("explain it like I'm 5" when asked), mobile-friendly
  pages, and the Bloomberg-terminal look of the site.

## How it runs (no PC needed)

- **Cloudflare Worker** (`cloudflare/worker.js`, deployed by `.github/workflows/deploy-worker.yml`):
  every 15 minutes, 9:30-16:00 New York time on weekdays, it dispatches the scan workflow. It
  also stores the state archive in the R2 bucket `options-dashboard-state` (latest plus one
  previous copy), behind the `STATE_TOKEN` key.
- **`.github/workflows/scan.yml`**: pulls the state, runs
  `python -m options_dashboard.cli --market-hours`, publishes `docs/`, pushes the state back.
  Run it by hand with `force=true` to scan outside market hours (no trades are made then).
- **`.github/workflows/research.yml`**: after each weekday's close, runs
  `python -m options_dashboard.longterm --publish` for the Invest page. Company reports are
  cached for a week, so a full refetch takes about 40 minutes and a normal run a few.
- Both share the `scan` concurrency group so they never touch the state at once.
- **The state is the source of truth in R2, not git.** `.state/`, `journal/`, `paper/` and
  `config.json` are git-ignored. `python -m options_dashboard.cloudstate pull|push` moves them
  (needs `STATE_URL` and `STATE_TOKEN`, from the environment or a local `.cloud.json`). `push`
  refuses unless a `pull` succeeded first in that run.
- The old Windows scheduled task "Options Dashboard Scanner" on the user's PC is **disabled on
  purpose**. Never re-enable it while the cloud runs, or every trade happens twice.
- The scanner commits `docs/` to `main` every 15 minutes in market hours: **`git pull --rebase`
  before committing**. `site.publish()` already retries after a rebase.
- Secrets live in GitHub Actions secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`,
  `DISPATCH_TOKEN` (a fine-grained PAT, Actions read/write, made Oct 2026 with about a year's
  expiry: if scans stop, check it first), `STATE_TOKEN`, `DISCORD_WEBHOOK` (the daily digest). The
  repo variable `STATE_URL` is the Worker's address.
- **Daily Discord digest** (`digest.py`): posted once each market day by the first scan at or after
  10:00 New York (the first with live quotes): regime, pictures of the sector and Mag 7 rotation
  maps, the next week's earnings with expectations and implied moves, contracts scoring 70+, and
  the Roth shortlist's buy-area changes. No strategy-account trades (the user asked to leave them
  out). Test it with the scan workflow's `digest_now` input.

- **0DTE lab** (`zerodte.py`, `daycall.py`): paper forward tests at real Cboe prices, run as a step before
  every scan. (1) The day's direction call for SPX and SPY, made at the open by a logistic model on
  pre-open data, posted to Discord at once, bought on confident days at about 9:30 prices (2x target or the
  close), scored daily alongside the dashboard's own SPY read. (2) The 3:30 PM end-of-day momentum rule.
  Both came from backtests (Dec 2023 - Oct 2026) whose small edges vanished if real option prices were 15%
  higher than modelled, so they are not to be presented as proven until the lab's real-price record says so.
  SPY showed no edge in either. The user caps 0DTE contracts at $150.

## Code map (`src/options_dashboard/`)

| Module | What it does |
|---|---|
| `cli.py` | Entry point for a scan: scan, journal, accounts, alerts, site |
| `config.py` | Every setting, including the strategy accounts; `config.json` overrides |
| `universe.py` | The liquid tickers scanned (screened, not capped) |
| `data.py` | Yahoo (paced, cached), Cboe chains (all expiries in one request, paced 2/s), market hours |
| `scanner.py` | Per-ticker read, contract selection and scoring; swing pool and a separate 0-14 day pool |
| `structure.py` | Auction, gamma/vanna/charm/DEX, Wyckoff, VPA reads and the composite |
| `flow.py` | Classifies option volume between scans as customer buys or sells; per-strike IV changes |
| `vol.py`, `avwap.py`, `sectors.py`, `seasonal.py`, `regime.py` | Volatility surface, AVWAPs, rotation (sectors and Mag 7), seasonality, market regime |
| `portfolio.py` | Simulated accounts: entry gate, structural stops, reviews, exits |
| `journal.py`, `learning.py`, `scorecard.py` | Follow every flagged contract 15 trading days, grade at 5/10/15, weekly weight tuning |
| `earnings.py` | Earnings outlook and plays |
| `longterm.py` | Invest page research: S&P 500 scores, buy areas, targets, cash machines, themes |
| `site.py` | Every page of the static site, the terminal styling, publishing |
| `cloudstate.py` | State to and from the Worker |
| `app.py` (repo root) | Local Streamlit dashboard (`start.bat` pulls the cloud state first) |

Hand-written research lives in `research/`: `themes.json` (themes traced to second- and
third-order beneficiaries) and earnings notes. Both are dated snapshots; refresh them when
asked, and update the date.

## Things learned the hard way

- Quotes are 15 minutes delayed: before 9:45 New York time they still show the pre-open book
  with zero bids. Never record or trade on zero or one-sided quotes.
- Yahoo rate-limits: go through `data.call` (paced, with backoff), and cache.
- Foreign listings often report in their home currency while the price is in dollars: skip any
  company whose `financialCurrency` differs from `currency`. Skip preferred shares too.
- Yahoo already restates share counts and EPS for splits. Do not adjust them again.
- Delayed quotes and a 15-minute cadence make simulated day-trade fills rough; say so.
- A company far below its own usual valuation may be a bargain or a value trap; flag it
  rather than trusting a mean-reversion target.
- Every chart on the site is drawn twice (desktop and phone widths, via `_both`) so it reads
  on a phone; keep new charts and tables usable at 360px wide.

## Working on it

- Python 3.12 with uv: `uv sync`, then `uv run python -m options_dashboard.cli` (add
  `--no-alert`; outside market hours it scans without trading).
- Checking a change: render the affected pages to a scratch folder and look at them at
  desktop and phone width before pushing. For scanner changes, scan a handful of liquid
  tickers rather than the whole universe.
- Cloud sessions may have limited network access, which blocks Yahoo, Cboe and the Worker.
  Code changes can still be made and pushed; the next scheduled cloud run picks them up.
- Commit messages explain what changed and why, in plain sentences.
