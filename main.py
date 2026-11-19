"""Leonida Week-One Forecast — make.fun gta6-trophy-rarity-forecast-01.

Forecasts every GTA VI trophy's Exophase earn rate at the case target as five
quantiles, from the house's reads up to the case checkpoint. Python 3.13,
standard library only, no network, fully deterministic.

Model, in one paragraph: trophy earn rates move on a log-time clock,
x = ln(1 + hours since 2026-11-19T00:00Z). For each trophy I fit two
recency-weighted least-squares lines on that clock — one for the logit of the
rate itself, one for the log of the earned count — and a third line for the
log of players tracked. The rate forecast blends the direct rate line with
earned/track (earned players divided by tracked players at the target), so a
jump in tracked players dilutes slow trophies instead of being ignored.
Slopes shrink toward a prior built from the trophy's tier and from the kind
of trophy its description names (story chapter, finale, mission, grind /
collectible). Medians are capped per tier and kind, and the spread combines
fit residual, a random-walk term in horizon days, and a refresh-jump term:
Exophase refreshes discretely, so a page that has sat unchanged for hours can
move every rate at once when it refreshes, and the quantiles must be wide
enough to admit that.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
ZVALUES = [-1.6448536269514722, -0.6744897501960817, 0.0,
           0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Slope prior (logit units per unit of log-time) by tier.
TIER_PRIOR = {"platinum": 0.0, "gold": 0.20, "silver": 0.30, "bronze": 0.40}
# Median caps (percent) by kind; platinum overrides everything.
KIND_CAP = {"story": 70.0, "finale": 50.0, "mission": 55.0,
            "grind": 35.0, "other": 60.0}
# Slope prior by kind.
KIND_PRIOR = {"story": 0.55, "finale": 0.35, "mission": 0.45,
              "grind": 0.12, "other": 0.25}

# Recency half-life for the weighted fits, in hours.
HALF_LIFE_H = 12.0
# How strongly a trophy's own fitted slope is pulled to its prior.
SHRINK_K = 2.5


def parse_time(text):
    """ISO-8601 text (with a trailing Z) to an aware datetime."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def log_clock(moment):
    """Log-time coordinate: ln(1 + hours since release midnight)."""
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def clamp_pct(value):
    return min(99.99, max(0.01, value))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(logit):
    if logit >= 0:
        p = 1.0 / (1.0 + math.exp(-logit))
    else:
        e = math.exp(logit)
        p = e / (1.0 + e)
    return min(100.0, max(0.01, 100.0 * p))


def round4(value):
    """Round to 4 decimals, halves away from zero, as a plain float/int."""
    scaled = value * 10000.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def classify_kind(trophy):
    """Sort a trophy into story / finale / mission / grind / other from the
    words of its own name and description — no facts from outside the case."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if "complete chapter" in text:
        return "story"
    if "final mission" in text:
        return "finale"
    if "100% completion" in text:
        return "grind"
    if ("complete " in text and any(w in text for w in
            ["250", "50 challenges", "8 mission", "3 collectible"])
            or "win 20" in text or "earn a gold medal" in text
            or "discover 100" in text or "explore all" in text
            or "purchase all" in text or "register 10" in text
            or "earn $" in text or "spend $" in text
            or "meditate at every" in text or "at 100 locations" in text
            or "hunt a unique" in text or "rob 5" in text
            or "rob 3" in text or "take down 5" in text):
        return "grind"
    if text.startswith("complete ") or "complete " in text:
        return "mission"
    return "other"


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, resid_var, n_eff, xbar, ybar): resid_var is the weighted
    mean squared residual, n_eff the weight sum normalised by the largest
    weight (how many fresh reads the fit is worth), and (xbar, ybar) the
    weighted centroid the line pivots on. With fewer than two distinct x
    values there is no slope to fit.
    """
    n = len(xs)
    if n == 0:
        return None
    wsum = sum(ws)
    if wsum <= 0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / wsum
    ybar = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12 or n < 2:
        resid = sum(w * (y - ybar) ** 2 for w, y in zip(ws, ys)) / wsum
        return (ybar, 0.0, resid, min(1.0, wsum / max(ws)), xbar, ybar)
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    inter = ybar - slope * xbar
    resid = sum(w * (y - (inter + slope * x)) ** 2
                for w, x, y in zip(ws, xs, ys)) / wsum
    return (inter, slope, resid, wsum / max(ws), xbar, ybar)


def forecast_trophy(trophy, reads, target_x, horizon_h, horizon_d,
                    tracked_line, flat_page):
    """Forecast one trophy. `reads` is a list of (clock_x, weight, rate, tracked)."""
    tier = trophy.get("tier", "bronze")
    kind = classify_kind(trophy)
    last_rate = reads[-1][2] if reads else None

    if last_rate is None:
        # No usable read at all: a deliberately vague, valid forecast.
        med = {"platinum": 0.5, "gold": 1.0, "silver": 2.0}.get(tier, 3.0)
        return quantiles(med, 1.5)

    if last_rate >= 99.99:
        # Already earned by (nearly) everyone tracked: it stays there.
        return quantiles(100.0, 0.05)

    prior = 0.5 * TIER_PRIOR.get(tier, 0.40) + 0.5 * KIND_PRIOR[kind]
    if tier == "platinum":
        prior = 0.0
    # A page that has not moved at all over a short horizon carries no
    # growth signal: the fitted slope is exactly zero and the prior must
    # not invent growth over the gap. The refresh below is the only move.
    frozen_short = flat_page and horizon_h <= 12.0

    xs = [r[0] for r in reads]
    ws = [r[1] for r in reads]

    # --- direct rate line: logit(rate) on log-time -------------------------
    rate_fit = weighted_line(xs, [logit_of(r[2]) for r in reads], ws)
    # --- earned-count line: ln(earned) on log-time ------------------------
    earned_ys = [math.log(max(0.5, r[2] * r[3] / 100.0)) for r in reads]
    earned_fit = weighted_line(xs, earned_ys, ws)

    candidates = []
    resid_var = 0.25
    if rate_fit is not None:
        _a, b, resid_var, n_eff, xbar, ybar = rate_fit
        b = max(-1.0, min(2.0, b))
        if not frozen_short:
            b = (n_eff * b + SHRINK_K * prior) / (n_eff + SHRINK_K)
        else:
            b = 0.0
        # Pivot on the data centroid, not on x = 0: the slope only acts
        # over the gap between the reads and the target.
        candidates.append(ybar + b * (target_x - xbar))
        resid_var = min(1.2, max(0.06, resid_var))
    if earned_fit is not None and tracked_line is not None:
        _a, b, _res, n_eff, xbar, ybar = earned_fit
        b = max(-0.3, min(2.0, b))
        if not frozen_short:
            b = (n_eff * b + SHRINK_K * prior) / (n_eff + SHRINK_K)
        else:
            b = 0.0
        earn_pred = ybar + b * (target_x - xbar)
        _ta, tb, _tr, _tn, txbar, tybar = tracked_line
        tb = max(0.0, min(2.0, tb))
        if not frozen_short:
            tb = (_tn * tb + SHRINK_K * 0.35) / (_tn + SHRINK_K)
        else:
            tb = 0.0
        track_pred = tybar + tb * (target_x - txbar)
        rate2 = 100.0 * math.exp(earn_pred) / max(1.0, math.exp(track_pred))
        candidates.append(logit_of(rate2))

    if candidates:
        med_logit = sum(candidates) / len(candidates)
    else:
        med_logit = logit_of(last_rate)

    # A page that has not moved at all is overdue a refresh: when Exophase
    # next refreshes, tracked jumps and every rate but the 100% one falls.
    # Launch night's refresh grew tracked ~6x while most trophies' earned
    # counts barely moved (the 100% trophy alone supplied nearly all the
    # new trophies earned), so a typical rate fell several-fold, not a
    # few percent. Shift the median down by most of that dilution and let
    # the spread (below) carry the rest of the refresh risk.
    if frozen_short:
        med_logit += -0.90 * (1.0 - last_rate / 100.0)

    median = inv_logit_pct(med_logit)

    # --- caps: never let a median run past what its tier/kind allows ------
    cap = 1.0 if tier == "platinum" else KIND_CAP[kind]
    if kind == "grind" and tier == "gold":
        cap = min(cap, 8.0)
    median = min(median, max(cap, last_rate))
    median = clamp_pct(median) if median < 100.0 else 100.0

    # --- spread: fit residual + random walk + refresh jump ----------------
    rw_var = (0.45 * math.sqrt(max(0.0, horizon_d))) ** 2
    if frozen_short:
        # An unseen refresh can move every rate several-fold at once; the
        # honest spread for a frozen page over a short horizon is ~1 logit.
        refresh_sd = 1.0
    elif horizon_h <= 6.0:
        refresh_sd = 0.50
    else:
        refresh_sd = 0.50 * math.sqrt(6.0 / horizon_h)
    spread = math.sqrt(resid_var + rw_var + refresh_sd ** 2)
    if last_rate < 1.0:
        spread *= 1.25
    elif last_rate < 3.0:
        spread *= 1.15
    elif last_rate < 10.0:
        spread *= 1.05
    spread = min(2.5, max(0.15, spread))
    # The tier/kind cap binds the whole band, not just the median: a
    # platinum cannot plausibly sit at 4% a week in, however wide the
    # spread. (Never below the last read rate, which may already exceed it.)
    band_cap = max(cap, last_rate)
    return [min(q, band_cap) for q in quantiles(median, spread)]


def quantiles(median, spread):
    """Five quantiles around a median percent, on the logit scale."""
    centre = 9.21 if median >= 100.0 else logit_of(median)
    out = []
    prev = None
    for z in ZVALUES:
        q = round4(min(100.0, max(0.01, inv_logit_pct(centre + z * spread))))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    trophies = case.get("trophies", [])
    history = case.get("history", [])
    horizon_min = float(case.get("horizon_minutes", 0) or 0)
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0
    target = parse_time(case["target"]) if case.get("target") else None
    target_x = log_clock(target) if target else log_clock(
        parse_time(case["at"])) if case.get("at") else 0.0

    # Recency weights, relative to the last read's clock (12 h half-life).
    weights = []
    if history:
        last_moment = parse_time(history[-1]["at"])
        for read in history:
            age_h = (last_moment - parse_time(read["at"])).total_seconds() / 3600.0
            weights.append(0.5 ** (max(0.0, age_h) / HALF_LIFE_H))

    # One global line for ln(players tracked) on log-time.
    track_xs, track_ys, track_ws = [], [], []
    for w, read in zip(weights, history):
        tracked = read.get("players_tracked")
        if tracked and tracked > 0:
            track_xs.append(log_clock(parse_time(read["at"])))
            track_ys.append(math.log(float(tracked)))
            track_ws.append(w)
    tracked_line = weighted_line(track_xs, track_ys, track_ws)

    # Is the whole page frozen? Every read identical to the first.
    flat_page = False
    if len(history) >= 2:
        first_rates = history[0].get("rates")
        flat_page = all(read.get("rates") == first_rates
                        and read.get("players_tracked") == history[0].get("players_tracked")
                        for read in history[1:])

    rows = []
    for idx, trophy in enumerate(trophies):
        reads = []
        for w, read in zip(weights, history):
            rates = read.get("rates") or []
            if idx >= len(rates):
                continue
            rate = rates[idx]
            if rate is None:
                continue
            tracked = read.get("players_tracked") or 0
            reads.append((log_clock(parse_time(read["at"])), w,
                          float(rate), float(tracked)))
        reads.sort(key=lambda r: r[0])
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(trophy, reads, target_x,
                                          horizon_h, horizon_d,
                                          tracked_line, flat_page)})
    return {"trophies": rows}


def fallback_case(case):
    """A valid forecast when a case is malformed: never crash, never be invalid."""
    rows = []
    for trophy in case.get("trophies", []):
        tier = trophy.get("tier", "bronze")
        med = {"platinum": 0.5, "gold": 1.0, "silver": 2.0}.get(tier, 3.0)
        rows.append({"id": trophy["id"], "q": quantiles(med, 1.5)})
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_case(case)
            except Exception:
                forecast = {"trophies": []}
        (OUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
