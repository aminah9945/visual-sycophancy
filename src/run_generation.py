"""Phase C3 / E1: generations without defence.

    python -m src.run_generation --exp E1 --split test --models all --samples 5
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src.llm.client import CFG, generate, spend
from src.metrics.grounded import load_table
from src.pipeline import make_record, run_parallel, tables, write_outputs
from src.vissyco.prompts import HYPS, messages_for


def gen_job(job):
    exp, tid, hyp, model, s, guard = job
    meta_s = tid.split("_")[0]
    msgs = messages_for(load_table(tid), meta_s, hyp, guard=guard)
    g = generate(model, msgs, sample_idx=s)
    return make_record(exp, tid, hyp, model, s, "B1" if guard else "none", msgs, g)


def models_arg(m: str) -> list[str]:
    return list(CFG["generators"]) if m == "all" else m.split(",")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E1")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--samples", type=int, default=CFG["decoding"]["samples"])
    ap.add_argument("--tables", default=None, help="comma list to restrict (pilot)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    tids = a.tables.split(",") if a.tables else tables(a.split)
    jobs = [(a.exp, t, h, m, s, False) for t in tids for h in HYPS for m in models_arg(a.models) for s in range(a.samples)]
    print(f"{a.exp}: {len(jobs)} generations; spend so far ${spend():.3f}")
    t0 = time.time()
    recs, errs = run_parallel(jobs, gen_job, a.workers, a.exp)
    recs.sort(key=lambda r: (r["model"], r["table_id"], r["hyp"], r["sample"]))
    d = write_outputs(a.exp, recs)
    df = pd.DataFrame(recs)
    print(f"done in {time.time()-t0:.0f}s, {len(recs)} records, {len(errs)} errors, spend ${spend():.3f} -> {d}")
    if len(df):
        print(df.groupby("model").agg(valid=("valid", "mean"), ei_ok=("ei_status", lambda s: s.isin(["ok", "ok_trend", "no_comparison"]).mean()),
                                      cong=("congruent", "mean"), fid=("fid", "mean")).round(3))


if __name__ == "__main__":
    main()
