#!/usr/bin/env python3
"""make.fun bounty pumpfun-top5-forecast-01: forecast the top 5 new Solana
(pump.fun) coins by market cap at each of the house's seven answer reads.

Model (stdlib only, deterministic, no network):
  * Parse every mark of /input/history.ndjson; track each mint's log10 market
    cap (in SOL) trajectory, venue (curve/migrating/pool), graduation and age.
  * For each answer read (from /input/task.json) run a seeded Monte Carlo:
      - Known coins: log-cap follows a rise-then-fade path. A coin's recent
        slope (shrunk, clamped) gives remaining growth g; after that it fades
        at a slow rate. Noise is Gaussian with sd growing as sqrt(hours).
        Pool (graduated) coins are more persistent than curve coins.
      - Newcomers: coins launched after the cut cannot be named, so they are
        simulated as anonymous challengers whose log-cap is drawn around the
        level new rockets reached in the history. They compete for the top 5
        exactly like known coins; their mass becomes the forecast's unnamed
        share, which is what scores when a newcomer actually takes a place.
  * Average the per-read distributions over the seven reads (one forecast is
    scored at every read, so the mixture is the honest single forecast).
  * Name a coin (or a size bin) at a place only when its probability beats
    the per-unit unnamed share it would otherwise receive; otherwise leaving
    it unnamed scores better. Floor named probabilities at 0.0001.
  * top5 is the greedy best-guess ordering (tie-break only).

Env: MAKEFUN_INPUT_DIR / INPUT_DIR (default /input),
     MAKEFUN_OUTPUT_DIR / OUTPUT_DIR (default /output).
"""

import heapq
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261112
NSIMS = 2000          # Monte Carlo draws per answer read
N_KNOWN = 60          # known coins simulated (largest by last cap)
N_NEW = 12            # anonymous newcomer challengers per read per sim
P_FLOOR = 0.0001
# --- calibrated dynamics (log10 cap in SOL) ---
POOL_FADE_PER_H = 0.04    # graduated coins fade this fast after their growth
CURVE_FADE_PER_H = 0.07   # curve coins fade faster
GROWTH_TAU_H = 1.5        # growth decays over ~this many hours
GROWTH_CAP = 0.45         # max remaining log10 growth credited from a trend
POOL_SIGMA_PER_SQRT_H = 0.22
CURVE_SIGMA_PER_SQRT_H = 0.25
NEWCOMER_SIGMA = 0.30
NEWCOMER_HEAT_PER_H = 0.02  # newcomer level drifts up as the day heats up


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def logcap_sol(market_cap_lamports):
    # log10 of the market cap in SOL: log10(lamports) - 9
    return math.log10(int(market_cap_lamports)) - 9.0


def main():
    input_dir = (os.environ.get("MAKEFUN_INPUT_DIR")
                 or os.environ.get("INPUT_DIR") or "/input")
    output_dir = (os.environ.get("MAKEFUN_OUTPUT_DIR")
                  or os.environ.get("OUTPUT_DIR") or "/output")

    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1  # 82
    reads = [parse_ts(r) for r in task["reads"]]
    out_name = task.get("output", "forecast.json")

    marks = []
    with open(os.path.join(input_dir, "history.ndjson"), "r",
              encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark")
    marks.sort(key=lambda m: parse_ts(m["at"]))
    cut_t = parse_ts(marks[-1]["at"])

    # ---- per-coin trajectories ----
    coins = {}  # mint -> dict
    for m in marks:
        t = parse_ts(m["at"])
        for r in m["rows"]:
            mint = r["mint"]
            c = coins.get(mint)
            if c is None:
                c = coins[mint] = {
                    "created": parse_ts(r["created_at"]),
                    "obs": [], "venue": r["venue"],
                    "complete": bool(r.get("complete")),
                    "mayhem": bool(r.get("mayhem")),
                }
            c["obs"].append((t, logcap_sol(r["market_cap_lamports"]),
                             r["venue"], bool(r.get("complete"))))
            c["venue"] = r["venue"]
            c["complete"] = c["complete"] or bool(r.get("complete"))

    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    if len(last_rows) < places:
        raise SystemExit("the last mark ranks fewer coins than the places")

    # ---- calibrate the newcomer level from the history ----
    # log-cap of the largest coin at the cut, and of the largest coin that
    # was young (<30 min old) at each mark: new rockets reach this level.
    top_log = logcap_sol(last_rows[0]["market_cap_lamports"])
    young_max = []
    for m in marks:
        t = parse_ts(m["at"])
        best = None
        for r in m["rows"]:
            age_h = (t - parse_ts(r["created_at"])) / 3600.0
            if age_h <= 0.5:
                lc = logcap_sol(r["market_cap_lamports"])
                best = lc if best is None else max(best, lc)
        if best is not None:
            young_max.append(best)
    if young_max:
        young_max.sort()
        new_level0 = young_max[len(young_max) // 2]  # median young max
    else:
        new_level0 = top_log
    # Newcomers are centred a touch below the current leader: they must beat
    # it, and the median young rocket sits just under the top.
    new_level0 = min(new_level0, top_log)

    # ---- known coins to simulate: largest by last cap ----
    # For each, fit: last log-cap, recent slope per hour (shrunk), venue.
    sim_coins = []
    for r in last_rows[:N_KNOWN]:
        mint = r["mint"]
        c = coins[mint]
        obs = c["obs"]
        x0 = obs[-1][1]
        # recent slope from the last up-to-3 observations, per hour
        slope = 0.0
        if len(obs) >= 2:
            (t1, x1, _, _), (t2, x2, _, _) = obs[-2], obs[-1]
            dt_h = max((t2 - t1) / 3600.0, 1e-6)
            slope = (x2 - x1) / dt_h
        if len(obs) >= 3:
            (t1, x1, _, _), (t3, x3, _, _) = obs[-3], obs[-1]
            dt_h = max((t3 - t1) / 3600.0, 1e-6)
            slope = 0.5 * slope + 0.5 * ((x3 - x1) / dt_h)
        slope = max(-0.15, min(0.20, slope))
        is_pool = (c["venue"] == "pool") or c["complete"]
        fade = POOL_FADE_PER_H if is_pool else CURVE_FADE_PER_H
        sigma = (POOL_SIGMA_PER_SQRT_H if is_pool
                 else CURVE_SIGMA_PER_SQRT_H)
        # remaining growth credited from the trend (shrunk: only half the
        # recent slope is trusted, and it decays over GROWTH_TAU_H)
        g = max(-0.10, min(GROWTH_CAP, 0.5 * slope * GROWTH_TAU_H))
        sim_coins.append({
            "mint": mint, "x0": x0, "g": g, "fade": fade, "sigma": sigma,
        })

    # ---- Monte Carlo over the answer reads ----
    rng = random.Random(SEED)
    # counts[place][mint] and new_counts[place]; size_hist[place][bin]
    counts = [dict() for _ in range(places)]
    new_counts = [0] * places
    size_hist = [[0] * n_bins for _ in range(places)]
    total_sims = 0

    for read_t in reads:
        h = max((read_t - cut_t) / 3600.0, 0.0)
        new_mu = new_level0 + NEWCOMER_HEAT_PER_H * h
        for _ in range(NSIMS):
            total_sims += 1
            entries = []  # (logcap, mint or None)
            for c in sim_coins:
                drift = (c["g"] * (1.0 - math.exp(-h / GROWTH_TAU_H))
                         - c["fade"] * h)
                noise = rng.gauss(0.0, c["sigma"] * math.sqrt(h)) if h > 0 \
                    else 0.0
                entries.append((c["x0"] + drift + noise, c["mint"]))
            for _ in range(N_NEW):
                entries.append((rng.gauss(new_mu, NEWCOMER_SIGMA), None))
            top = heapq.nlargest(places, entries, key=lambda e: e[0])
            for k, (lc, mint) in enumerate(top):
                if mint is None:
                    new_counts[k] += 1
                else:
                    d = counts[k]
                    d[mint] = d.get(mint, 0) + 1
                # size bin for this place
                b = bin_of(lc, edges)
                size_hist[k][b] += 1

    # ---- average over reads (already pooled: every read ran NSIMS sims) ----
    # Per-place known-coin probabilities and the unnamed (newcomer + unlisted)
    # mass. A coin is named only when its probability beats the per-coin
    # unnamed share it would otherwise receive.
    probabilities = []
    chart_n = int(task.get("chart_size", 100))
    for k in range(places):
        d = counts[k]
        # candidate probabilities, highest first
        cand = sorted(((p / total_sims, mint) for mint, p in d.items()),
                      key=lambda t: -t[0])
        named = []
        named_sum = 0.0
        for p, mint in cand:
            unnamed_n = chart_n - len(named)
            if unnamed_n <= 0:
                break
            share = (1.0 - named_sum) / unnamed_n
            if p <= share:
                break
            if p < P_FLOOR:
                break
            named.append((mint, p))
            named_sum += p
            if len(named) >= 100:
                break
        # renormalise guard: never exceed 1 in total
        if named_sum > 1.0:
            scale = 1.0 / named_sum
            named = [(m, p * scale) for m, p in named]
        for mint, p in named:
            probabilities.append({"mint": mint, "place": k + 1,
                                  "p": math.floor(p * 1e6) / 1e6})

    # ---- sizes: average the per-read histograms, same inclusion rule ----
    sizes = []
    for k in range(places):
        hist = size_hist[k]
        probs = [(c / total_sims, b) for b, c in enumerate(hist) if c > 0]
        probs.sort(key=lambda t: -t[0])
        named = []
        named_sum = 0.0
        for p, b in probs:
            unnamed_n = n_bins - len(named)
            if unnamed_n <= 0:
                break
            share = (1.0 - named_sum) / unnamed_n
            if p <= share:
                break
            if p < P_FLOOR:
                break
            named.append((b, p))
            named_sum += p
        if named_sum > 1.0:
            scale = 1.0 / named_sum
            named = [(b, p * scale) for b, p in named]
        for b, p in named:
            sizes.append({"place": k + 1, "bin": b,
                          "p": math.floor(p * 1e6) / 1e6})

    # ---- top5: greedy best-guess ordering (tie-break only) ----
    # expected place score per coin, then assign distinct coins to places
    # 1..5 by highest probability at each place.
    best_at = {}
    for k in range(places):
        d = counts[k]
        if not d:
            continue
        mint, _ = max(d.items(), key=lambda kv: kv[1])
        best_at[k] = mint
    top5 = []
    used = set()
    for k in range(places):
        # first choice for this place, else the highest-probability coin
        # overall not yet used
        cand_mint = best_at.get(k)
        if cand_mint is None or cand_mint in used:
            rest = sorted(
                ((sum(counts[j].get(m, 0) for j in range(places)), m)
                 for m in set().union(*[set(d) for d in counts])),
                key=lambda t: -t[0])
            cand_mint = next((m for _, m in rest if m not in used), None)
        if cand_mint is None:
            break
        top5.append(cand_mint)
        used.add(cand_mint)
    # fill from the last mark's order if short (should not happen)
    for r in last_rows:
        if len(top5) >= places:
            break
        if r["mint"] not in used:
            top5.append(r["mint"])
            used.add(r["mint"])
    top5 = top5[:places]

    forecast = {"probabilities": probabilities, "top5": top5,
                "sizes": sizes}
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")
    return out_path


def bin_of(lc, edges):
    # bins: 0 below edges[0]; i (1..len-1) from edges[i-1] to edges[i];
    # len(edges) at/above edges[-1]. 82 bins for 81 edges.
    if lc < edges[0]:
        return 0
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= lc:
            lo = mid + 1
        else:
            hi = mid
    # lo = number of edges <= lc -> bin index
    return lo


if __name__ == "__main__":
    path = main()
    print("wrote", path)
