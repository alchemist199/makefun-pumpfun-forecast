# pumpfun-top5-forecast-01 — Python forecaster

Forecasts the top 5 new pump.fun coins by market cap at each of the house's
seven answer reads. `main.py` (Python 3.13, standard library only) reads
`/input/history.ndjson` + `/input/task.json` and writes `/output/forecast.json`
(`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` override the paths).

## The model

1. **Parse the history.** Every mark's rows become per-coin trajectories of
   log10 market cap (in SOL), with venue (`curve` / `migrating` / `pool`),
   graduation and age. The 60 largest coins at the last mark are simulated;
   the rest can only matter through the unnamed share.
2. **Monte Carlo, seeded (fixed seed, deterministic).** For each of the seven
   answer reads (times read from `task.json`, never hardcoded), 2000 draws:
   - *Known coins:* log-cap follows a rise-then-fade path. A coin's recent
     slope (shrunk, clamped) gives remaining growth that decays over ~1.5 h;
     after that it fades (graduated coins slower than curve coins). Gaussian
     noise with sd growing as sqrt(hours to the read).
   - *Newcomers:* coins launched after the cut cannot be named, so they are
     simulated as 12 anonymous challengers per read, their log-cap drawn
     around the level new rockets reached in the history (median max cap of
     coins <30 min old), drifting up slightly as the day heats up. They
     compete for the top 5 like known coins; their mass becomes the
     forecast's unnamed share — which is exactly what scores when a newcomer
     actually takes a place (3 of the top 5 at the practice's last read were
     coins the history never named).
3. **Average over reads.** One forecast is scored at every read, so the
   mixture over the seven reads is the honest single forecast.
4. **Honest naming rule.** A coin (or size bin) is named at a place only when
   its probability beats the per-unit unnamed share it would otherwise
   receive; otherwise leaving it unnamed scores better. Named probabilities
   are floored to 6 decimals so no place's sum can exceed 1. `top5` is the
   greedy best-guess ordering (tie-break only).

## Why this beats persistence

The house baseline bets the last snapshot's order holds. In the practice
history the top 5 turns over almost completely within an hour and new coins
rocket to #1 within minutes of launch, so persistence is wrong most of the
time. This forecaster models the fade, the graduation/pump dynamics and the
newcomer flood explicitly, and keeps a large honest unnamed share (about
0.5–0.7 per place in the practice run, more at the lower places over the
final's 7–13 h horizon) instead of betting everything on today's leaders.

## Checks (practice data)

- `MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=<out> python3 main.py`
  runs in ~1 s; two runs write byte-identical output.
- Practice forecast: 6,831 bytes, exactly the keys `probabilities`, `top5`,
  `sizes`; top5 is 5 distinct mints, all in the history; every place's
  probability and size sums are ≤ 1; every named mint appears in the history.
- Order score at the practice's last read (whose top 5 the rules publish):
  **−21.86**, vs the house baseline's −23.21 on the same read. The baseline's
  mean order score over all seven practice reads is −21.38, and its total
  practice score is −40.41; this entry's honest sizes and larger unnamed
  share are designed to improve on both parts.
- A synthetic 66-mark final-shape history also runs in ~1 s with valid output.
