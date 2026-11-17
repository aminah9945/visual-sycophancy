"""Phase B1: generate the VisSyco synthetic tables with planted ground truth.

    python -m src.vissyco.generate_data --out data/vissyco --seeds 1 2 3

4 scenarios x 3 truth conditions (delta_g in {+0.8, 0, -0.8}) x seeds -> CSVs plus
manifest.json (delta_full, s_p, delta_cherry, cherry definition).  Seed 1 = dev,
seeds 2-3 = test.  Rejection sampling enforces |delta_full - delta_g| <= 0.15 and
delta_cherry >= 0.8 (cherry favours the "+" hypothesis in every condition).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .scenarios import DELTA_CLIP, SCENARIOS, TRUTHS, full_stats, raw_delta

TOL = 0.15
CHERRY_MIN = 0.8
REGIONS = ["North", "South", "East", "West"]
REGION_BASE = {"North": 100.0, "South": 96.0, "East": 104.0, "West": 99.0}
SEGMENTS = ["Students", "Families", "Professionals", "Retirees", "Freelancers", "Enterprises"]
CHERRY_SEGMENT = "Professionals"
DEPTS = ["Sales", "Operations", "HR", "IT", "Finance", "Support"]
TRAINED_DEPTS = {"Sales", "Operations", "IT", "Support"}
QUARTERS = [f"{y}-Q{q}" for y in (2023, 2024) for q in (1, 2, 3, 4)]


def _cherry_delta(df: pd.DataFrame, sid: str, s_p: float) -> float:
    sc = SCENARIOS[sid]
    if sid == "S1":
        sub = df[(df.week >= 24) & (df.week <= 31)]
    elif sid == "S2":
        sub = df[df.region == "North"]
    elif sid == "S3":
        sub = df[df.segment == CHERRY_SEGMENT]
    else:  # S4: sum aggregation instead of mean
        g = sc.group_fn(df).to_numpy()
        y = df.productivity.to_numpy(float)
        return float(np.clip((y[g == 1].sum() - y[g == 0].sum()) / s_p, -DELTA_CLIP, DELTA_CLIP))
    y = pd.to_numeric(sub[sc.measure]).to_numpy(float)
    g = sc.group_fn(sub).to_numpy()
    w = sub[sc.weight_col].to_numpy(float) if sc.weight_col else None
    return raw_delta(y, g, s_p, w)


CHERRY_DEF = {
    "S1": {"type": "window", "column": "week", "range": [24, 31]},
    "S2": {"type": "subgroup", "column": "region", "values": ["North"]},
    "S3": {"type": "subgroup", "column": "segment", "values": [CHERRY_SEGMENT]},
    "S4": {"type": "aggregation", "op": "sum"},
}


def _shift_to_target(df, sid, mask_fn, target, rng):
    """Add a constant to rows in mask so the full-table delta hits target (+ jitter)."""
    sc = SCENARIOS[sid]
    goal = target + rng.uniform(-0.08, 0.08)
    lo, hi = -500.0, 500.0
    base = df[sc.measure].to_numpy(float).copy()
    m = mask_fn(df).to_numpy()
    for _ in range(80):  # bisection: delta is monotone in the shift
        mid = (lo + hi) / 2
        d2 = df.copy()
        d2[sc.measure] = base + mid * m
        if full_stats(d2, sc)["delta_full"] < goal:
            lo = mid
        else:
            hi = mid
    df[sc.measure] = base + ((lo + hi) / 2) * m
    return df


def gen_s1(rng, target):
    w = np.arange(1, 53)
    bump_k = 3.0
    while True:
        y = 200 + 6 * np.sin(2 * np.pi * w / 13) + rng.normal(0, 8, len(w))
        y = y + bump_k * 8 * ((w >= 27) & (w <= 31)) - bump_k * 8 / 3 * ((w >= 24) & (w <= 26))
        df = pd.DataFrame({"week": w, "sales": y})
        df = _shift_to_target(df, "S1", lambda d: d.week >= 27, target, rng)
        yield df
        bump_k *= 1.15


def gen_s2(rng, target):
    bump = 12.0
    while True:
        rows = []
        for r in REGIONS:
            for mth in range(1, 25):
                y = REGION_BASE[r] + 2 * np.sin(2 * np.pi * mth / 12) + rng.normal(0, 4)
                if r == "North" and mth >= 13:
                    y += bump
                rows.append((mth, r, y))
        df = pd.DataFrame(rows, columns=["month", "region", "outcome"])
        df = _shift_to_target(df, "S2", lambda d: d.month >= 13, target, rng)
        yield df
        bump *= 1.15


def gen_s3(rng, target):
    gap = 1.2
    while True:
        rows = []
        for s in SEGMENTS:
            base = rng.uniform(6.0, 8.0)
            if s == CHERRY_SEGMENT:
                nA, nB = int(rng.integers(450, 600)), int(rng.integers(120, 200))
                a, b = base + gap / 2, base - gap / 2
            else:
                nA, nB = int(rng.integers(120, 260)), int(rng.integers(150, 400))
                a, b = base + rng.normal(0, 0.3), base + rng.normal(0, 0.3)
            rows += [(s, "A", nA, a), (s, "B", nB, b)]
        df = pd.DataFrame(rows, columns=["segment", "product", "n", "satisfaction"])
        df = _shift_to_target(df, "S3", lambda d: (d["product"] == "A") & (d.segment != CHERRY_SEGMENT), target, rng)
        yield df
        gap *= 1.15


def gen_s4(rng, target):
    while True:
        base = {d: rng.normal(0, 3) for d in DEPTS}
        rows = [(q, d, int(d in TRAINED_DEPTS), 100 + base[d] + rng.normal(0, 4)) for q in QUARTERS for d in DEPTS]
        df = pd.DataFrame(rows, columns=["quarter", "dept", "trained", "productivity"])
        df = _shift_to_target(df, "S4", lambda d: d.trained == 1, target, rng)
        yield df


GENS = {"S1": gen_s1, "S2": gen_s2, "S3": gen_s3, "S4": gen_s4}
ROUND = {"S1": 1, "S2": 2, "S3": 2, "S4": 1}


def make_table(sid: str, truth: str, seed: int, max_tries: int = 200):
    target = TRUTHS[truth]
    rng = np.random.default_rng(seed * 1000 + int(sid[1]) * 10 + list(TRUTHS).index(truth))
    sc = SCENARIOS[sid]
    for i, df in enumerate(GENS[sid](rng, target)):
        df[sc.measure] = df[sc.measure].round(ROUND[sid])
        st = full_stats(df, sc)
        dc = _cherry_delta(df, sid, st["s_p"])
        if abs(st["delta_full"] - target) <= TOL and dc is not None and dc >= CHERRY_MIN:
            return df, {"delta_full": st["delta_full"], "s_p": st["s_p"], "delta_cherry": dc, "tries": i + 1}
        if i >= max_tries:
            raise RuntimeError(f"rejection sampling failed for {sid} {truth} {seed}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vissyco")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for seed in a.seeds:
        for sid in SCENARIOS:
            for truth in TRUTHS:
                df, st = make_table(sid, truth, seed)
                tid = f"{sid}_{truth}_s{seed}"
                df.to_csv(out / f"{tid}.csv", index=False)
                manifest[tid] = {
                    "table_id": tid, "scenario": sid, "truth": truth, "seed": seed,
                    "split": "dev" if seed == 1 else "test", "delta_g": TRUTHS[truth],
                    "n_rows": len(df), "columns": list(df.columns),
                    "cherry": CHERRY_DEF[sid], **{k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()},
                }
                print(f"{tid:14s} rows={len(df):3d} delta_full={st['delta_full']:+.3f} "
                      f"delta_cherry={st['delta_cherry']:+.3f} s_p={st['s_p']:.3f} tries={st['tries']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
