#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster for make.fun bounty
roblox-up-and-coming-forecast-01 (program/v1 harness).

Reads /input/history.ndjson (hourly-ish snapshots of Roblox's Up-and-Coming
chart: each game's rank, universe_id, name, playing) and /input/task.json,
and writes /output/forecast.json:
  {"probabilities": [{"universe_id", "place", "p"}], "top10": [10 ids]}

Model (pure standard library, deterministic, no network, no RNG):
  For each game, its rank over the next 72 hours is modelled as a Gaussian
  diffusion: rank(h) ~ Normal(r0 + v*h, s0^2 + sigma^2*h), where r0 is the
  rank projected to the end of history, v is the recent rank velocity
  (ranks/hour, exponentially weighted, shrunk toward 0) and sigma is the
  game's recent hourly rank volatility (shrunk toward the global value).
  A small per-hour exit hazard covers games vanishing from the chart.
  P(game at place k) is the average over h=1..72 of that Gaussian's mass on
  the rank bin for k. Each place's named probabilities are then scaled so
  the place sums to 1 with an "unnamed" share E(k): the measured share of
  top-10 slots at that place held by games that were not on the chart 72
  hours earlier (true new entrants / fast climbers the history cannot name).
  Rows weaker than the unnamed per-game probability are left unnamed.

Measured on the public archive (Kenny-Dhub/roblox-snapshots, 2026-10-30 to
2026-11-13, 52 reads): hourly rank-change std ~2.1 ranks; top-10 overlap
after 24h 6.4/10, after 72h 3.4/10; ~31% of top-10 slots are held by games
not on the chart 72h earlier. There is NO hard 28-day age-out in the data:
games 30-140 days old stay on the chart, so no age cutoff is applied.

Env overrides (as in the kit's starter): MAKEFUN_INPUT_DIR, MAKEFUN_OUTPUT_DIR.
Defaults: /input and /output.
"""

import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Tuned constants (tuned on the practice backtest; see backtest.py).
# ---------------------------------------------------------------------------
S0 = 2.0            # initial rank uncertainty (ranks)
SIGMA_GLOBAL = 2.12  # global hourly rank volatility (ranks/hour), measured
SIGMA_FLOOR = 1.2
SIGMA_CAP = 5.0
VEL_HALF_LIFE_H = 33.0   # exponential decay for velocity fit
VEL_SHRINK_N = 2.0       # velocity shrinkage strength
VEL_CAP = 0.5            # |velocity| cap, ranks/hour
HAZARD_PER_HOUR = 0.0015  # mild exit hazard (game vanishes from the chart)
STALE_HAZARD_PER_HOUR = 0.01  # extra hazard if last seen > 24h ago
ENTRANT_PRIOR = 0.10     # prior unnamed share per place (measured, candidate-based)
ENTRANT_PRIOR_N = 3.0    # pseudo-count for per-place entrant shrinkage
TEMP = 0.8               # diffusion-width multiplier (calibration)
PERSIST_MIX = 0.20        # weight of the persistence (baseline-like) mix
VEL_W = 0.2               # weight on measured rank velocity (it mean-reverts)
CCU_BETA = -8.0           # ranks/hour per unit log-CCU/hour slope
DRIFT = 0.24              # mean downward rank drift, ranks/hour (chart churn)
H_CAP = 48                # variance-saturation horizon, hours (mean reversion)
MIN_NAMED_P = 0.001      # smallest probability worth naming
HOURS = 72               # forecast horizon in hours


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_history(input_dir):
    path = os.path.join(input_dir, "history.ndjson")
    snaps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    return snaps


def parse_time(s):
    # ISO 8601 UTC, e.g. 2026-11-07T20:53:19.000Z
    return _to_hours(s)


def _to_hours(s):
    from datetime import datetime
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.timestamp() / 3600.0


def build_series(snaps):
    """Per-game observations: {uid: {'name':, 'obs': [(t_hours, rank, playing)]}}.
    Also returns snapshot times and chart sizes."""
    series = {}
    times = []
    for snap in snaps:
        t = _to_hours(snap["at"])
        times.append(t)
        for r in snap.get("up_and_coming") or []:
            u = r["universe_id"]
            e = series.setdefault(u, {"name": r.get("name"), "obs": []})
            e["obs"].append((t, r["rank"], r.get("playing")))
            if r.get("name"):
                e["name"] = r["name"]
    return series, times


def fit_ccu_slope(obs, t0):
    """Exponentially-weighted slope of log(playing) vs time (last ~48h).
    CCU growth is a leading indicator: a game whose player count is growing
    climbs the chart even before its rank has moved. Measured on the archive:
    corr(CCU slope, 24h rank change) = -0.26, independent of rank velocity
    (corr -0.08); regression: future rank change ~ +5.8 + 0.17*vel*dt
    - 5.3*ccu*dt (R^2 0.12 vs 0.06 vel-only). Returns (slope, effective_n)."""
    pts = [(t, math.log(max(p, 1))) for (t, _r, p) in obs
           if p is not None and t0 - t <= 48]
    if len(pts) < 2:
        return 0.0, 0.0
    wts = [math.exp(-(t0 - t) * math.log(2) / VEL_HALF_LIFE_H) for (t, _y) in pts]
    sw = sum(wts)
    if sw <= 0:
        return 0.0, 0.0
    mt = sum(w * t for w, (t, _y) in zip(wts, pts)) / sw
    my = sum(w * y for w, (_t, y) in zip(wts, pts)) / sw
    num = sum(w * (t - mt) * (y - my) for w, (t, y) in zip(wts, pts))
    den = sum(w * (t - mt) ** 2 for w, (t, _y) in zip(wts, pts))
    if den <= 0:
        return 0.0, 0.0
    slope = num / den
    slope *= sw / (sw + VEL_SHRINK_N)
    return max(-0.2, min(0.2, slope)), sw


def fit_velocity(obs, t0):
    """Exponentially-weighted linear fit of rank vs time (last ~5 days).
    Returns (velocity ranks/hour, effective_n)."""
    pts = [(t, rank) for (t, rank, _p) in obs if t0 - t <= 5 * 24]
    if len(pts) < 2:
        return 0.0, 0.0
    wts = [math.exp(-(t0 - t) * math.log(2) / VEL_HALF_LIFE_H) for (t, _r) in pts]
    sw = sum(wts)
    if sw <= 0:
        return 0.0, 0.0
    mt = sum(w * t for w, (t, _r) in zip(wts, pts)) / sw
    mr = sum(w * r for w, (_t, r) in zip(wts, pts)) / sw
    num = sum(w * (t - mt) * (r - mr) for w, (t, r) in zip(wts, pts))
    den = sum(w * (t - mt) ** 2 for w, (t, _r) in zip(wts, pts))
    if den <= 0:
        return 0.0, 0.0
    v = num / den
    n_eff = sw  # ~ number of (decayed) observations
    v *= n_eff / (n_eff + VEL_SHRINK_N)
    v = max(-VEL_CAP, min(VEL_CAP, v))
    return v, n_eff


def fit_sigma(obs, t0, global_sigma):
    """Hourly rank volatility from consecutive observations (last ~5 days),
    shrunk toward the global value."""
    vals = []
    pts = [(t, rank) for (t, rank, _p) in obs if t0 - t <= 5 * 24]
    for (t1, r1), (t2, r2) in zip(pts, pts[1:]):
        dt = t2 - t1
        if dt > 0:
            vals.append(((r2 - r1) / math.sqrt(dt)) ** 2)
    if not vals:
        return global_sigma
    n = len(vals)
    var = (sum(vals) + 8 * global_sigma ** 2) / (n + 8)
    return max(SIGMA_FLOOR, min(SIGMA_CAP, math.sqrt(var)))


def measure_entrants(snaps, series):
    """Per-place share of top-10 slots held by games OUTSIDE the candidate
    set (games seen in the 7 days before the read with last rank <= 70) --
    i.e. the mass the forecaster genuinely cannot name. Measured over the
    given history, using only snapshots before each read (no lookahead).
    The right quantity for the unnamed share: a game that debuted in the
    chart a few days ago IS a candidate, so the older "not in chart 72h
    earlier" definition overstates it. Measured on the archive this is
    ~0.06-0.10 overall, much smaller than the ~0.31 72h-entrant rate."""
    counts = {k: [0, 0] for k in range(1, 11)}
    t_first = _to_hours(snaps[0]["at"])
    for i, snap in enumerate(snaps):
        t = _to_hours(snap["at"])
        if t - t_first < 3 * 24:
            continue  # need enough prior history for a meaningful read
        last = {}
        for s in snaps[:i]:
            ts = _to_hours(s["at"])
            if t - ts > 7 * 24:
                continue
            for r in s.get("up_and_coming") or []:
                last[r["universe_id"]] = (ts, r["rank"])
        cand = {u for u, (_tt, rr) in last.items() if rr <= 70}
        for r in sorted(snap.get("up_and_coming") or [], key=lambda x: x["rank"])[:10]:
            k = r["rank"]
            counts[k][1] += 1
            if r["universe_id"] not in cand:
                counts[k][0] += 1
    out = []
    for k in range(1, 11):
        e, n = counts[k]
        out.append((e + ENTRANT_PRIOR_N * ENTRANT_PRIOR) / (n + ENTRANT_PRIOR_N) if n else ENTRANT_PRIOR)
    return out


def measure_global_sigma(snaps):
    """Global hourly rank volatility from all consecutive snapshot pairs."""
    vals = []
    for s1, s2 in zip(snaps, snaps[1:]):
        dt = _to_hours(s2["at"]) - _to_hours(s1["at"])
        if dt <= 0:
            continue
        m1 = {r["universe_id"]: r["rank"] for r in s1.get("up_and_coming") or []}
        m2 = {r["universe_id"]: r["rank"] for r in s2.get("up_and_coming") or []}
        for u in set(m1) & set(m2):
            if m1[u] <= 40:
                vals.append(((m2[u] - m1[u]) / math.sqrt(dt)) ** 2)
    if not vals:
        return SIGMA_GLOBAL
    return max(1.0, min(4.0, math.sqrt(sum(vals) / len(vals))))


def forecast(history_snaps, places=10):
    series, _times = build_series(history_snaps)
    t0 = _to_hours(history_snaps[-1]["at"])
    last_chart = sorted(history_snaps[-1].get("up_and_coming") or [], key=lambda r: r["rank"])
    last_rank = {r["universe_id"]: r["rank"] for r in last_chart}
    chart_size = len(last_chart) or 95
    global_sigma = measure_global_sigma(history_snaps)
    entrant = measure_entrants(history_snaps, series)

    # Candidate games: seen in the last 7 days and recently ranked <= 70,
    # or in the last snapshot's top 70.
    candidates = []
    for uid, e in series.items():
        obs = e["obs"]
        t_last, r_last, _p = obs[-1]
        if t0 - t_last > 7 * 24:
            continue
        if r_last > 70 and uid not in last_rank:
            continue
        if r_last > 70 and last_rank.get(uid, 999) > 70:
            continue
        candidates.append(uid)

    # Per-game diffusion distribution P(g at place k), averaged over h=1..72.
    dist = {}  # uid -> list P[1..10]
    for uid in candidates:
        e = series[uid]
        obs = e["obs"]
        v, _n = fit_velocity(obs, t0)
        ccu, _nc = fit_ccu_slope(obs, t0)
        v_eff = VEL_W * v + CCU_BETA * ccu + DRIFT
        v_eff = max(-VEL_CAP, min(VEL_CAP, v_eff))
        sigma = fit_sigma(obs, t0, global_sigma)
        t_last, r_last, _p = obs[-1]
        r0 = r_last + v_eff * max(0.0, t0 - t_last)
        r0 = max(1.0, r0)
        stale = (t0 - t_last) > 24
        hazard = HAZARD_PER_HOUR + (STALE_HAZARD_PER_HOUR if stale else 0.0)
        probs = [0.0] * (places + 1)  # 1-indexed
        for h in range(1, HOURS + 1):
            mu = r0 + v_eff * h
            # variance saturates (mean reversion) instead of growing linearly
            s = TEMP * math.sqrt(S0 * S0 + sigma * sigma * min(h, H_CAP))
            surv = math.exp(-hazard * h)
            for k in range(1, places + 1):
                lo = 0.5 if k == 1 else k - 0.5
                hi = k + 0.5
                p = surv * (norm_cdf((hi - mu) / s) - norm_cdf((lo - mu) / s))
                probs[k] += p / HOURS
        dist[uid] = probs

    # Persistence mix: 0.5 on the game at each place as of the last snapshot
    # (the house baseline's shape), blended in lightly for calibration.
    persist = {}
    for k in range(1, places + 1):
        if k - 1 < len(last_chart):
            persist.setdefault(last_chart[k - 1]["universe_id"], {})[k] = 0.5
        if k == 1 and len(last_chart) > 1:
            persist.setdefault(last_chart[1]["universe_id"], {})[k] = 0.3
        if k > 1:
            if k - 2 < len(last_chart):
                persist.setdefault(last_chart[k - 2]["universe_id"], {})[k] = 0.15
            if k < len(last_chart):
                persist.setdefault(last_chart[k]["universe_id"], {})[k] = 0.15

    rows = []
    for k in range(1, places + 1):
        # blend diffusion with persistence
        pk = {}
        for uid, probs in dist.items():
            p = probs[k] * (1.0 - PERSIST_MIX)
            if uid in persist and k in persist[uid]:
                p += PERSIST_MIX * persist[uid][k]
            if p > 0:
                pk[uid] = p
        total = sum(pk.values())
        e_share = entrant[k - 1]
        if total <= 0:
            continue
        # scale named so they sum to (1 - entrant share)
        scale = (1.0 - e_share) / total
        named = {uid: p * scale for uid, p in pk.items() if p * scale >= MIN_NAMED_P}
        # prune rows weaker than the unnamed per-game probability, twice
        for _ in range(2):
            unnamed = 1.0 - sum(named.values())
            n_unnamed = max(chart_size - len(named), 1)
            thr = unnamed / n_unnamed
            dropped = {uid: p for uid, p in named.items() if p < thr}
            if not dropped:
                break
            for uid in dropped:
                del named[uid]
        for uid, p in sorted(named.items(), key=lambda kv: -kv[1]):
            rows.append({"universe_id": uid, "place": k, "p": round(min(p, 0.95), 6)})

    # top10: greedy distinct assignment by max probability per place
    taken = set()
    top10 = []
    for k in range(1, places + 1):
        best = None
        best_p = -1.0
        for r in rows:
            if r["place"] == k and r["universe_id"] not in taken and r["p"] > best_p:
                best = r["universe_id"]
                best_p = r["p"]
        if best is None:
            # fall back to the last chart's game at this place
            if k - 1 < len(last_chart):
                best = last_chart[k - 1]["universe_id"]
        if best is not None and best not in taken:
            taken.add(best)
            top10.append(best)
    # fill any gaps from the last chart order
    for r in last_chart:
        if len(top10) >= places:
            break
        if r["universe_id"] not in taken:
            taken.add(r["universe_id"])
            top10.append(r["universe_id"])
    return {"probabilities": rows, "top10": top10[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    snaps = load_history(input_dir)
    fc = forecast(snaps, places=places)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(fc) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
