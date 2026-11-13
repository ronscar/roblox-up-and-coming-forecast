# Roblox Up-and-Coming forecast — make.fun bounty entry

Program for the make.fun bounty **roblox-up-and-coming-forecast-01**
("Predict Roblox's Up-and-Coming top 10", $500). It reads the history of
Roblox's Up-and-Coming chart and forecasts the top 10 (a probability for
each game at each place, plus the top 10 in order) for the 72 hours after
entries close.

## How it works

`main.py` (Python 3, standard library only, deterministic, no network) reads
`/input/history.ndjson` and `/input/task.json` and writes
`/output/forecast.json`.

Model (see the header of `main.py` for the full description):

- Each game's rank over the next 72 h is a Gaussian diffusion:
  rank(h) ~ Normal(r0 + v_eff*h, s0^2 + sigma^2*min(h, 48)), where r0 is the
  rank projected to the end of history, sigma is the game's recent hourly
  rank volatility (shrunk toward the global value, ~2.1 ranks/hour), and the
  variance saturates (mean reversion) instead of growing linearly.
- Effective drift v_eff = 0.2 * (rank velocity) - 5.0 * (log-CCU slope) + 0.10.
  Player-count (CCU) growth is a leading indicator: on the public archive
  it predicts the next 24 h of rank change independently of rank velocity
  (corr -0.26, regression R^2 0.12 vs 0.06 for velocity alone). The +0.10 is
  the mean downward drift of established games (chart churn).
- A small per-hour exit hazard covers games vanishing from the chart.
- P(game at place k) is the average over h = 1..72 of that Gaussian's mass
  on the rank bin for k, blended 80/20 with a persistence distribution
  (the house baseline's shape).
- Each place's named probabilities are scaled so the place sums to 1 with
  an "unnamed" share: the measured share of top-10 slots held by games
  outside the candidate set (games not seen in the last 7 days with last
  rank <= 70), about 0.06-0.10 overall. Rows weaker than the unnamed
  per-game probability are left unnamed.
- There is deliberately NO 28-day age cutoff: the archive shows games
  30-140 days old still on the chart (e.g. Ragdoll Soccer, 30 days old,
  ranked #6 on 2026-11-13), so the brief's "published in the last 28 days"
  does not act as a hard cutoff in the data.

## Backtest results (exact make.fun log score, `score.py`)

Scorer verified against the kit: it reproduces the house baseline's
practice score exactly (-43.512039, 6.4 top-10 hits, 53.5 agreement).

| window | train | reads | house baseline | this program |
|---|---|---|---|---|
| practice (2026-11-07T23:00Z - 2026-11-10T23:00Z) | kit history (34 snaps, ends 2026-11-07 20:53Z) | 10 archive reads | -43.51 | **-34.45** (6.4 top-10 hits) |
| recent (2026-11-10 23:23Z - 2026-11-13 02:45Z) | archive <= 2026-11-10 23:23Z (44 snaps) | 9 reads (8 archive + 1 live) | -46.77 | **-35.95** (5.1 top-10 hits) |

House model ("superbot (practice)") practice score: -37.2. Best entrants so
far (make.fun API): ~-33.6. So on the practice window this program beats
the house model by ~2.8 nats and sits ~0.9 nats behind the best entrant.
Run `python3 backtest.py` to reproduce. Final tuned parameters are in
`main.py` (TEMP=0.8, PERSIST_MIX=0.20, VEL_W=0.2, CCU_BETA=-8.0,
DRIFT=0.24, H_CAP=48); the tuning grid and measurements are in `NOTES.md`.

Local pre-submit check (`makefun check`, run 2026-11-13): passes
competition_open, snapshot (3 files, 21,190 bytes), license_osi (MIT) and
entrypoint (main.py -> python3.13). It fails only on transcripts_present
(no transcripts bundled yet - see entry steps) and the at-submission
checks (public repo, owner, created-after-open, pushed commit), which need
the entrant's own GitHub account. `makefun grade-local` is not available
for this competition (scorer and reference data are server-only); the
practice score above is computed with `score.py`, which reproduces the
kit's stated house-baseline practice score exactly (-43.512039, 6.4 hits,
53.5 agreement), so the scoring rule is verified.

## How to run

```bash
python3 main.py            # reads /input, writes /output/forecast.json
MAKEFUN_INPUT_DIR=... MAKEFUN_OUTPUT_DIR=... python3 main.py
```

Requirements: Python 3.8+ (uses only the standard library). Runs in well
under 300 s on one CPU.

## Data sources

- Bounty kit: https://make.fun/c/roblox-up-and-coming-forecast-01/kit
  (RULES.txt, competition.yaml, cli/makefun.mjs, input/history.ndjson).
- Chart history used for backtesting: public archive
  https://github.com/Kenny-Dhub/roblox-snapshots (MIT), converted to the
  kit's snapshot format in `data/archive.ndjson` (52 reads,
  2026-10-30T07:36:30Z to 2026-11-13T00:25:48Z).
- Live chart endpoint (verified working 2026-11-13):
  `https://apis.roblox.com/explore-api/v1/get-sort-content?sortId=up-and-coming&sessionId=<uuid>&device=computer&country=us`
  and game metadata `https://games.roblox.com/v1/games?universeIds=<ids>`.

## Entry steps (what the entrant must do)

1. Create a make.fun account (a Solana keypair is the account; a person
   signs in on the site with a wallet, or via GitHub/Google/Discord).
2. Create a public GitHub repo owned by your own GitHub account, created
   after the competition opened (2026-11-11 06:00 UTC), under an
   OSI-approved license (this repo uses MIT), with `main.py` in the root.
3. Bundle the AI session transcripts that built it:
   `node cli/makefun.mjs bundle --repo <owner>/<repo> --out transcripts.tar.gz --redact "<your full name>"`
4. Check the entry:
   `node cli/makefun.mjs check --kit . --repo <owner>/<repo> --transcripts transcripts.tar.gz`
5. Submit before the deadline (2026-11-13 06:00 UTC = 13:00 WIB):
   `node cli/makefun.mjs submit --competition roblox-up-and-coming-forecast-01 --repo-url https://github.com/<owner>/<repo> --commit <sha> --title "<name>" --bundle transcripts.tar.gz --keypair wallet.json`

Deadline: **2026-11-13 06:00 UTC (13:00 WIB)**. At most 5 submissions a day,
20 in all; the last submission before the deadline is the final entry.
