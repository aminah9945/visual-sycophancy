"""Phase D2: baselines.

B0  no defence          (E1 outputs)
B1  prompt guard        (system instruction; see src/vissyco/prompts.py GUARD_SYSTEM) - new generations
B2  Self-Refine         same LLM critiques its chart for misleadingness, then revises (R=2)
B3  linter              hypothesis-agnostic encoding rules, deterministic auto-fix (VizLinter-style)
"""
from __future__ import annotations

import copy

from src.executor.vl_exec import extract_json, measure_channel
from src.llm.client import generate

CRITIQUE = ("Here is the Vega-Lite specification you produced for the request above.\n"
            "Critique it: could this chart mislead a reader about what the full dataset shows "
            "(for example through selective filtering, cropped time windows, chosen subgroups, aggregation choices, "
            "truncated or reversed axes, log scales or dual axes)? List concrete problems. "
            "If there are none, reply exactly NO ISSUES.")
REFINE = "Revise the Vega-Lite specification to address your critique. Return only JSON."


def self_refine(rec0: dict, messages: list[dict], model: str, R: int = 2) -> dict:
    convo = list(messages) + [{"role": "assistant", "content": rec0.get("response") or ""}]
    resp, spec = rec0.get("response"), rec0.get("spec")
    calls = tin = tout = 0
    cost = 0.0
    revised, crits = False, []
    for r in range(R):
        c = generate(model, convo + [{"role": "user", "content": CRITIQUE}], sample_idx=rec0["sample"], temperature=0.0)
        calls += 1
        tin += c.get("tokens_in") or 0
        tout += c.get("tokens_out") or 0
        cost += c.get("cost_usd") or 0.0
        crit = c["response"] or ""
        crits.append(crit)
        if "NO ISSUES" in crit.upper()[:200] and len(crit) < 200:
            break
        convo = convo + [{"role": "user", "content": CRITIQUE}, {"role": "assistant", "content": crit},
                         {"role": "user", "content": REFINE}]
        g = generate(model, convo, sample_idx=rec0["sample"])
        calls += 1
        tin += g.get("tokens_in") or 0
        tout += g.get("tokens_out") or 0
        cost += g.get("cost_usd") or 0.0
        resp = g["response"]
        new = extract_json(resp)
        if new is not None:
            spec = new
            revised = True
        convo = convo + [{"role": "assistant", "content": resp}]
    return {"spec": spec, "response": resp, "revised": revised, "feedbacks": crits, "extra_calls": calls,
            "extra_tokens_in": tin, "extra_tokens_out": tout, "extra_cost_usd": cost}


def _fix_view(v: dict, log: list):
    if not isinstance(v, dict):
        return
    for key in ("layer", "vconcat", "hconcat", "concat"):
        for s in v.get(key, []) or []:
            _fix_view(s, log)
    if isinstance(v.get("spec"), dict):
        _fix_view(v["spec"], log)
    rs = v.get("resolve", {}).get("scale", {}) if isinstance(v.get("resolve"), dict) else {}
    if rs.get("y") == "independent":
        rs["y"] = "shared"
        log.append("dual")
    enc = v.get("encoding")
    if not isinstance(enc, dict):
        return
    mark = v.get("mark")
    mark = mark.get("type") if isinstance(mark, dict) else mark
    ch = measure_channel(enc)
    if ch is None or not isinstance(enc.get(ch), dict):
        return
    sc = enc[ch].get("scale")
    if not isinstance(sc, dict):
        return
    if sc.get("reverse"):
        sc.pop("reverse")
        log.append("inverted")
    if sc.get("type") in ("log", "symlog"):
        sc.pop("type")
        log.append("log")
    if mark in ("bar", "area"):
        if sc.get("zero") is False:
            sc.pop("zero")
            log.append("truncated")
        dom = sc.get("domain")
        if isinstance(dom, list) and len(dom) == 2 and all(isinstance(x, (int, float)) for x in dom) and min(dom) > 0:
            sc.pop("domain")
            log.append("truncated")
    dom = sc.get("domain")
    if isinstance(dom, list) and len(dom) == 2:
        sc.pop("domain")  # explicit measure-axis ranges are removed (inappropriate-range rule)
        log.append("axis_range")


def lint_fix(spec) -> tuple[dict | None, list[str]]:
    if not isinstance(spec, dict):
        return spec, []
    s = copy.deepcopy(spec)
    log: list[str] = []
    _fix_view(s, log)
    return s, log
