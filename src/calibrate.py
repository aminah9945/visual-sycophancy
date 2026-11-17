"""Phase D3: threshold calibration on the DEV split only (paper Eq. 8).

    python -m src.run_generation --exp DEV --split dev        # base generations (once)
    python -m src.calibrate --split dev --eps 0.05

tau* = argmin_tau VSR_dev(A_tau o G)  s.t.  FRR_dev(tau) <= eps; ties -> larger tau (fewer revisions).
Writes runs/CAL/grid.csv and freezes tau* in configs/experiments.yaml.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import yaml

from src.metrics.grounded import vsr_units
from src.pipeline import RUNS, read_records, run_parallel
from src.run_defenses import cgca_variant

ROOT = Path(__file__).resolve().parents[1]
TAU_EI = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
TAU_LF = (1.1, 1.25, 1.5, 2.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--base", default="DEV")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    assert a.split == "dev", "never tune on the test split"
    base = [r for r in read_records(a.base) if r["split"] == "dev"]
    for r in base:
        r["exp_out"] = "CAL"
    rows = []
    for te, tl in itertools.product(TAU_EI, TAU_LF):
        recs, _ = run_parallel(base, cgca_variant("CGCA", te, tl, a.R), a.workers, f"tau=({te},{tl})")
        df = pd.DataFrame(recs)
        u = vsr_units(df)
        neu = df[(df.hyp == "neutral") & df.valid.astype(bool)]
        frr = float(neu["flagged_r0"].astype(float).mean()) if len(neu) else float("nan")
        rows.append({"tau_ei": te, "tau_lf": tl, "VSR_dev": float(u["diff"].mean()) if len(u) else float("nan"),
                     "FRR_dev": frr, "revised_rate": float(df["revised"].astype(float).mean()),
                     "extra_calls": float(df["extra_calls"].mean())})
        print(rows[-1])
    g = pd.DataFrame(rows)
    (RUNS / "CAL").mkdir(parents=True, exist_ok=True)
    g.to_csv(RUNS / "CAL" / "grid.csv", index=False)
    ok = g[g.FRR_dev <= a.eps]
    if ok.empty:
        print(f"no tau satisfies FRR <= {a.eps}; choosing the lowest-FRR setting")
        best = g.sort_values(["FRR_dev", "VSR_dev"]).iloc[0]
    else:
        best = ok.sort_values(["VSR_dev", "tau_ei", "tau_lf"], ascending=[True, False, False]).iloc[0]
    cfgp = ROOT / "configs" / "experiments.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["cgca"].update({"tau_ei": float(best.tau_ei), "tau_lf": float(best.tau_lf), "calibrated": True,
                        "calibration": {"VSR_dev": round(float(best.VSR_dev), 4), "FRR_dev": round(float(best.FRR_dev), 4), "eps": a.eps}})
    cfgp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=200))
    print("tau* =", dict(best))


if __name__ == "__main__":
    main()
