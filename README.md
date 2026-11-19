[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ImRevoltexee/Makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ImRevoltexee/Makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

# Leonida Week-One Forecast

Forecasts every GTA VI trophy's Exophase earn rate at 2026-11-26T00:00:00Z —
one week after launch — as five quantiles per trophy, from the house's reads
up to the case checkpoint.

## Model

- Log-time clock `x = ln(1 + hours since 2026-11-19T00:00Z)`. Per trophy,
  recency-weighted least squares of logit(rate) on `x`, plus a second line
  for log earned-counts divided by a global log players-tracked line — so a
  jump in tracked players dilutes slow trophies instead of being ignored.
- Slopes shrink toward a prior from the trophy's tier and from the kind its
  own description names (story chapter, finale, mission, grind/collectible).
- Medians capped per tier/kind (platinum ≤ 1%, story bronze ≤ 70%).
- Spread = fit residual + random walk in horizon days + a refresh-jump term.
  Exophase refreshes discretely: a page frozen for hours can move every
  rate several-fold at once, so a frozen page over a short horizon gets a
  wide spread and a downward-shifted median.

## Run offline

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input dir, writes `/output/<id>.out` per case.
Python 3.13, standard library only, no network, deterministic.
