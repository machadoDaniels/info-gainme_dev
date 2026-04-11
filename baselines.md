# Optimal baseline (worst-case, `ceil(N/2)`)

- **Maximum theoretical IG (per turn)** = **1 bit**: a yes/no question removes at most half the uncertainty over a uniform space; the ceiling is reached with a 50/50 partition.
- **Total IG** from *N* candidates down to 1 = **log₂(N)** (same as initial *H*; telescoping sum).
- **Per-turn IG** in the baseline below does not hit 1 on every step: with odd *N* on the worst branch, some turns have IG below 1. 
- **Average IG per turn** ≈ **log₂(N) ÷ turns** on this path (8 turns for the *N* in the table).

Values from `scripts/compute_optimal_baseline.py` (same convention as `src.entropy.Entropy`):

## Objects (158) — `data/objects/objects_full.csv`

(`python scripts/compute_optimal_baseline.py --n 158 --json`)

| Metric               | Value |
|----------------------|-------|
| Initial *H* (bits)   | 7.303780748177103 |
| Total IG (bits)      | 7.303780748177103 |
| Avg. IG/turn         | 0.9129725935221379 |
| Turns                | 8 |

## Geo (160) — e.g. `data/geo/top_160_pop_cities.csv`

(`python scripts/compute_optimal_baseline.py --n 160 --json`)

| Metric               | Value |
|----------------------|-------|
| Initial *H* (bits)   | 7.321928094887363 |
| Total IG (bits)      | 7.321928094887363 |
| Avg. IG/turn         | 0.9152410118609203 |
| Turns                | 8 |

## Diseases (160) — e.g. `data/diseases/diseases_160.csv`

(same *N* as Geo 160 — `python scripts/compute_optimal_baseline.py --n 160 --json`)

| Metric               | Value |
|----------------------|-------|
| Initial *H* (bits)   | 7.321928094887363 |
| Total IG (bits)      | 7.321928094887363 |
| Avg. IG/turn         | 0.9152410118609203 |
| Turns                | 8 |

### Reference formulas

```text
Optimal total IG  =  log₂(N)
Avg. IG per turn   =  log₂(N) / turns     (this worst-case baseline)
```

With *N* equally likely candidates, initial *H* = log₂(N) bits.
