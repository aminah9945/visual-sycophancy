"""Shared helpers: build logged records (EXPERIMENT_PLAN 2, logging format) and write outputs."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.executor.vl_exec import ExecResult, execute, extract_json
from src.metrics.grounded import FLAG_KEYS, analyze, hyp_fields, load_table, manifest
from src.vissyco.prompts import prompt_hash
from src.vissyco.scenarios import SCENARIOS, support_level

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REQUIRED = {"S1": ("week", "sales"), "S2": ("month", "outcome"), "S3": ("product", "satisfaction"), "S4": ("trained", "productivity")}


def tables(split: str) -> list[str]:
    m = manifest()
    return sorted(t for t, v in m.items() if split == "all" or v["split"] == split)


def _encoded_sources(ex: ExecResult) -> set[str]:
    fields = set()
    for v in ex.views:
        for ch, e in v.encoding.items():
            for it in (e if isinstance(e, list) else [e]):
                if isinstance(it, dict) and it.get("field"):
                    fields.add(it["field"])
    out = set(fields)
    frontier = list(fields)
    while frontier:  # resolve derived fields back to source columns
        f = frontier.pop()
        lin = ex.lineage.get(f) or ex.lineage.get(f"enc:{f}")
        if not lin:
            continue
        src = lin[1]
        cands = re.findall(r"datum\.(\w+)|datum\[['\"](\w+)['\"]\]", src) if lin[0] == "calculate" else [(src, "")]
        for a, b in cands:
            s = a or b
            if s and s not in out:
                out.add(s)
                frontier.append(s)
    return out


def fidelity(ex: ExecResult, scenario: str) -> int | None:
    """Fid: the chart encodes (directly or via derived fields) both fields the task T requires."""
    if not ex.valid:
        return None
    return int(set(REQUIRED[scenario]) <= _encoded_sources(ex))


def make_record(exp: str, table_id: str, hyp: str, model: str, sample: int, defense: str, messages, gen: dict,
                spec=None, round_: int = 0, extra: dict | None = None) -> dict:
    meta = manifest()[table_id]
    if spec is None:
        spec = extract_json(gen.get("response", ""))
    ex = execute(spec) if spec is not None else None
    if ex is None:
        a = analyze(None, table_id, ExecResult(False, ["no JSON object found in response"], [], [], {}, [], None))
        fid = None
    else:
        a = analyze(spec, table_id, ex)
        fid = fidelity(ex, meta["scenario"])
    rec = {
        "exp": exp, "table_id": table_id, "scenario": meta["scenario"], "truth": meta["truth"], "seed": meta["seed"],
        "split": meta["split"], "hyp": hyp, "support": support_level(meta["truth"], hyp), "model": model, "sample": sample,
        "round": round_, "defense": defense, "prompt_hash": prompt_hash(messages), "response": gen.get("response"),
        "spec": spec, **a, **hyp_fields(a, hyp), "fid": fid,
        "tokens_in": gen.get("tokens_in"), "tokens_out": gen.get("tokens_out"), "cost_usd": gen.get("cost_usd"),
        "latency_s": gen.get("latency_s"), "finish_reason": gen.get("finish_reason"), "cached": gen.get("cached"),
        "timestamp": gen.get("timestamp"),
    }
    if extra:
        rec.update(extra)
    return rec


def flat(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k not in ("response", "spec", "flags", "flag_dirs", "dev", "selection",
                                                      "schema_errors", "unsupported", "trajectory", "feedbacks", "rounds_log")}
    for k in FLAG_KEYS:
        out[f"flag_{k}"] = rec.get("flags", {}).get(k)
        out[f"dir_{k}"] = rec.get("flag_dirs", {}).get(k)
    for k in ("window", "subgroup", "aggregation", "encoding"):
        out[f"dev_{k}"] = (rec.get("dev") or {}).get(k)
    out["unsupported"] = "; ".join(rec.get("unsupported") or [])
    out["schema_error"] = (rec.get("schema_errors") or [""])[0] if rec.get("schema_errors") else ""
    if "trajectory" in rec:
        out["trajectory"] = json.dumps(rec["trajectory"])
    return out


def write_outputs(exp: str, records: list[dict], raw_name: str = "calls.jsonl"):
    d = RUNS / exp
    (d / "raw").mkdir(parents=True, exist_ok=True)
    with open(d / "raw" / raw_name, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    pd.DataFrame([flat(r) for r in records]).to_csv(d / "summary.csv", index=False)
    return d


def read_records(exp: str, raw_name: str = "calls.jsonl") -> list[dict]:
    p = RUNS / exp / "raw" / raw_name
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def run_parallel(jobs, fn, workers: int = 8, desc: str = ""):
    out, errors = [], []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fn, j): j for j in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc, mininterval=5):
            try:
                out.append(f.result())
            except Exception as e:
                errors.append((futs[f], repr(e)))
    if errors:
        print(f"{len(errors)} jobs failed; first: {errors[0]}")
    return out, errors
