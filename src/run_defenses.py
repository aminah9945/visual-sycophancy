"""Phase D4 / E2 and ablations.

    python -m src.run_defenses --exp E2 --split test --defenses B0,B1,B2,B3,CGCA
    python -m src.run_defenses --exp ABL --split test --ablations A1,A2,A3,A4,A5,A6 --models qwen3-30b,gpt-4o-mini

All post-hoc defences start from the same E1 generation (paired design -> McNemar).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.cgca.audit import run_cgca
from src.cgca.baselines import lint_fix, self_refine
from src.llm.client import spend
from src.metrics.grounded import load_table
from src.pipeline import RUNS, make_record, read_records, run_parallel, write_outputs
from src.run_generation import gen_job, models_arg
from src.vissyco.prompts import messages_for

ROOT = Path(__file__).resolve().parents[1]


def taus():
    cfg = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
    c = cfg["cgca"]
    return float(c["tau_ei"]), float(c["tau_lf"]), int(c.get("R", 2))


def _msgs(rec):
    return messages_for(load_table(rec["table_id"]), rec["scenario"], rec["hyp"])


def _gen0(rec):
    return {k: rec.get(k) for k in ("response", "tokens_in", "tokens_out", "cost_usd", "latency_s", "finish_reason", "cached", "timestamp")}


def cgca_variant(name: str, tau_ei: float, tau_lf: float, R: int, **kw):
    def fn(rec):
        out = run_cgca(rec, _msgs(rec), rec["model"], tau_ei, tau_lf, R=R, **kw)
        g = _gen0(rec)
        g["response"] = out["response"]
        extra = {k: out[k] for k in ("trajectory", "violations", "feedbacks", "revised", "flagged_r0", "augmented",
                                     "rounds_used", "extra_calls", "extra_tokens_in", "extra_tokens_out", "extra_cost_usd")}
        extra.update({"tau_ei": tau_ei, "tau_lf": tau_lf, "R": R})
        return make_record(rec["exp_out"], rec["table_id"], rec["hyp"], rec["model"], rec["sample"], name, _msgs(rec), g,
                           spec=out["spec"], extra=extra)
    return fn


def b0(rec):
    r = dict(rec)
    r.update({"exp": rec["exp_out"], "defense": "B0", "revised": False, "extra_calls": 0})
    r.pop("exp_out", None)
    return r


def b2(rec):
    out = self_refine(rec, _msgs(rec), rec["model"])
    g = _gen0(rec)
    g["response"] = out["response"]
    extra = {k: out[k] for k in ("revised", "feedbacks", "extra_calls", "extra_tokens_in", "extra_tokens_out", "extra_cost_usd")}
    return make_record(rec["exp_out"], rec["table_id"], rec["hyp"], rec["model"], rec["sample"], "B2", _msgs(rec), g,
                       spec=out["spec"], extra=extra)


def b3(rec):
    spec, log = lint_fix(rec.get("spec"))
    return make_record(rec["exp_out"], rec["table_id"], rec["hyp"], rec["model"], rec["sample"], "B3", _msgs(rec), _gen0(rec),
                       spec=spec, extra={"revised": bool(log), "lint_fixes": log, "extra_calls": 0})


def b1_jobs(base, exp):
    seen = {(r["table_id"], r["hyp"], r["model"], r["sample"]) for r in base}
    return [(exp, t, h, m, s, True) for (t, h, m, s) in sorted(seen)]


def b1(job):
    r = gen_job(job)
    r["defense"] = "B1"
    r["revised"] = False
    r["extra_calls"] = 0
    return r


def ablation_defs(names, te, tl, R):
    d = {}
    for a in names:
        if a == "A1":
            d["A1-EIonly"] = cgca_variant("A1-EIonly", te, tl, R, use_flags=False)
            d["A1-flagsonly"] = cgca_variant("A1-flagsonly", te, tl, R, use_ei=False)
        elif a == "A2":
            for r in (0, 1):
                d[f"A2-R{r}"] = cgca_variant(f"A2-R{r}", te, tl, r)
        elif a == "A3":
            d["A3-generic"] = cgca_variant("A3-generic", te, tl, R, feedback="generic")
        elif a == "A4":
            d["A4-agnostic"] = cgca_variant("A4-agnostic", te, tl, R, agnostic=True)
        elif a == "A5":
            d["A5-noaug"] = cgca_variant("A5-noaug", te, tl, R, do_augment=False)
        elif a == "A6":
            for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                if abs(t - te) > 1e-9:
                    d[f"A6-tau{t}"] = cgca_variant(f"A6-tau{t}", t, tl, R)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2")
    ap.add_argument("--base", default=None, help="base generation run (default E1 for test, DEV for dev)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--defenses", default="")
    ap.add_argument("--ablations", default="")
    ap.add_argument("--models", default="all")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    base_exp = a.base or ("E1" if a.split == "test" else "DEV")
    models = set(models_arg(a.models))
    base = [r for r in read_records(base_exp) if r["model"] in models and (a.split == "all" or r["split"] == a.split)]
    for r in base:
        r["exp_out"] = a.exp
    te, tl, R = taus()
    fns = {}
    for d in filter(None, a.defenses.split(",")):
        fns[d] = {"B0": b0, "B2": b2, "B3": b3, "CGCA": cgca_variant("CGCA", te, tl, R)}.get(d, "B1" if d == "B1" else None)
    fns.update(ablation_defs(filter(None, a.ablations.split(",")), te, tl, R))
    if any(d in ("A1", "A2", "A3", "A4", "A5", "A6") for d in a.ablations.split(",")) and "CGCA" not in fns:
        fns["CGCA"] = cgca_variant("CGCA", te, tl, R)
        fns["B0"] = b0
    out = []
    print(f"{a.exp}: base={base_exp} n={len(base)} defenses={list(fns)} tau_ei={te} tau_lf={tl} R={R} spend=${spend():.3f}")
    for name, fn in fns.items():
        if fn == "B1":
            recs, _ = run_parallel(b1_jobs(base, a.exp), b1, a.workers, "B1")
        else:
            recs, _ = run_parallel(base, fn, a.workers, name)
        out += recs
        print(f"  {name}: {len(recs)} records, spend ${spend():.3f}")
    out.sort(key=lambda r: (r["defense"], r["model"], r["table_id"], r["hyp"], r["sample"]))
    print("->", write_outputs(a.exp, out))


if __name__ == "__main__":
    main()
