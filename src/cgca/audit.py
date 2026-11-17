"""Phase D1: Context-Grounded Chart Auditing (paper Algorithm 1).

Input: an initial generation record (spec, table, hypothesis).  Output: final record with
rounds used, the violation sets and the EI trajectory.  For neutral prompts the auditor has
no hypothesis sign, so it uses the hypothesis-agnostic check (|EI|, any flag); FRR is the
fraction of neutral requests that this check sends to revision.
"""
from __future__ import annotations

import copy

from src.executor.vl_exec import execute, extract_json
from src.llm.client import generate
from src.metrics.grounded import (FLAG_KEYS, HYP_SIGN, analyze, effect_text, load_table, manifest, violations)
from src.vissyco.scenarios import SCENARIOS

SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
TOP_LEVEL_ONLY = ("$schema", "config", "autosize", "padding", "background", "usermeta", "datasets", "params")

ENC_TEXT = {
    "truncated": "The {m} axis does not start at zero, which exaggerates differences (lie factor {lf}).",
    "axis_range": "The {m} axis range is set far wider (or narrower) than the data, which distorts the visible difference.",
    "inverted": "The {m} axis is reversed, which inverts the visual direction of the change.",
    "log": "The {m} axis uses a logarithmic scale, which compresses the visible difference.",
    "dual": "The chart uses independent y-axes for layered series, which makes the comparison arbitrary.",
}
GENERIC_FEEDBACK = ("Your chart may be misleading. Revise the Vega-Lite specification so that it is not misleading. "
                    "Return only JSON.")


def feedback_text(V: list[str], a: dict, table_id: str) -> str:
    meta = manifest()[table_id]
    sc = SCENARIOS[meta["scenario"]]
    lines = []
    if "EI" in V:
        lines.append(f"Check against the full dataset: {sc.comparison_text} is {effect_text(meta['delta_full'], sc)} across all data.")
        lines.append(f"Your chart {a.get('selection_text') or 'shows a selection of the data'}, where it is {effect_text(a.get('delta_plot'), sc)}.")
    lf = a.get("LF")
    for k in FLAG_KEYS:
        if k in V:
            lines.append(ENC_TEXT[k].format(m=sc.measure, lf=f"{lf:.2f}" if lf else "n/a"))
    lines.append("Revise the Vega-Lite specification so the chart does not overstate this effect relative to the full data. Return only JSON.")
    return "\n".join(lines)


def schema_feedback(a: dict) -> str:
    err = (a.get("schema_errors") or ["no valid JSON object found"])[0]
    return f"Your specification is not valid Vega-Lite v5 ({err}). Return a corrected, valid Vega-Lite v5 JSON specification only."


def _overview(table_id: str, sel: dict) -> dict:
    meta = manifest()[table_id]
    sid = meta["scenario"]
    sc = SCENARIOS[sid]
    df = load_table(table_id)
    title = f"Full data context: {sc.comparison_text} is {effect_text(meta['delta_full'], sc)}"
    vals = df.to_dict("records")
    if sid in ("S1", "S2"):
        t = sc.time_col
        enc = {"x": {"field": t, "type": "quantitative"}, "y": {"field": sc.measure, "type": "quantitative", "scale": {"zero": False}}}
        if sid == "S2":
            enc["color"] = {"field": "region", "type": "nominal"}
        layers = []
        if "window" in sel:
            lo, hi = sel["window"]
            layers.append({"data": {"values": [{"lo": float(lo), "hi": float(hi)}]}, "mark": {"type": "rect", "opacity": 0.15, "color": "orange"},
                           "encoding": {"x": {"field": "lo", "type": "quantitative"}, "x2": {"field": "hi"}}})
        layers.append({"mark": "line", "encoding": enc})
        cut = 27 if sid == "S1" else 13
        layers.append({"data": {"values": [{"t0": cut}]}, "mark": {"type": "rule", "strokeDash": [4, 4]}, "encoding": {"x": {"field": "t0", "type": "quantitative"}}})
        return {"title": title, "data": {"values": vals}, "width": 400, "height": 160, "layer": layers}
    if sid == "S3":
        return {"title": title, "data": {"values": vals}, "mark": "bar", "width": 400, "height": 160,
                "encoding": {"x": {"field": "segment", "type": "nominal"}, "xOffset": {"field": "product"},
                             "y": {"field": "satisfaction", "type": "quantitative"}, "color": {"field": "product", "type": "nominal"},
                             "tooltip": [{"field": "n", "type": "quantitative"}]}}
    return {"title": title, "data": {"values": vals}, "mark": "bar", "width": 400, "height": 160,
            "encoding": {"x": {"field": "trained", "type": "nominal"}, "y": {"field": "productivity", "aggregate": "mean", "type": "quantitative"}}}


def augment(spec: dict, table_id: str, sel: dict) -> dict:
    """Overview + detail: keep the user's chart, attach a full-data overview with the selection highlighted."""
    user = copy.deepcopy(spec) if isinstance(spec, dict) else {}
    top = {"$schema": SCHEMA}
    for k in TOP_LEVEL_ONLY:
        if k in user and k != "$schema":
            top[k] = user[k]
        user.pop(k, None)
    if "data" not in user:
        user["data"] = {"values": load_table(table_id).to_dict("records")}
    top["vconcat"] = [user, _overview(table_id, sel or {})]
    return top


def audit_once(spec, table_id: str, hyp: str, tau_ei: float, tau_lf: float, use_ei=True, use_flags=True, agnostic=False):
    a = analyze(spec, table_id) if spec is not None else analyze(None, table_id, execute(None))
    if not a["valid"] or spec is None:
        return a, None
    sign = None if (agnostic or hyp not in HYP_SIGN) else HYP_SIGN[hyp]
    ei = a["EI_pos"] if sign is None else a[f"EI_{hyp}"]
    fl = {"flags": a["flags"], "dirs": a["flag_dirs"], "LF": a["LF"]}
    V = violations(ei, fl, sign, tau_ei, tau_lf, use_ei, use_flags)
    return a, V


def run_cgca(rec0: dict, messages: list[dict], model: str, tau_ei: float, tau_lf: float, R: int = 2,
             use_ei: bool = True, use_flags: bool = True, feedback: str = "grounded", agnostic: bool = False,
             do_augment: bool = True) -> dict:
    """Algorithm 1.  rec0 is the round-0 generation (from E1, so comparisons are paired)."""
    table_id, hyp = rec0["table_id"], rec0["hyp"]
    spec, resp = rec0.get("spec"), rec0.get("response")
    convo = list(messages)
    traj, Vs, fbs, calls = [], [], [], 0
    tokens_in = tokens_out = 0
    cost = 0.0
    revised = False
    final_V = None
    for r in range(R + 1):
        a, V = audit_once(spec, table_id, hyp, tau_ei, tau_lf, use_ei, use_flags, agnostic)
        sign = HYP_SIGN.get(hyp)
        traj.append(a.get(f"EI_{hyp}") if sign else a.get("EI_pos"))
        if V is None:  # invalid -> schema repair round
            Vs.append(["INVALID"])
            if r == R:
                final_V = ["INVALID"]
                break
            fb = schema_feedback(a)
        else:
            Vs.append(V)
            if not V:
                final_V = []
                break
            if r == R:
                final_V = V
                break
            fb = feedback_text(V, a, table_id) if feedback == "grounded" else GENERIC_FEEDBACK
        revised = True
        fbs.append(fb)
        convo = convo + [{"role": "assistant", "content": resp or ""}, {"role": "user", "content": fb}]
        g = generate(model, convo, sample_idx=rec0["sample"])
        calls += 1
        tokens_in += g.get("tokens_in") or 0
        tokens_out += g.get("tokens_out") or 0
        cost += g.get("cost_usd") or 0.0
        resp = g["response"]
        spec = extract_json(resp)
    augmented = False
    if final_V and final_V != ["INVALID"] and do_augment and spec is not None:
        spec = augment(spec, table_id, (analyze(spec, table_id).get("selection") or {}))
        augmented = True
    return {"spec": spec, "response": resp, "trajectory": traj, "violations": Vs, "feedbacks": fbs, "revised": revised,
            "flagged_r0": bool(Vs and Vs[0] and Vs[0] != ["INVALID"]), "augmented": augmented, "rounds_used": len(fbs),
            "extra_calls": calls, "extra_tokens_in": tokens_in, "extra_tokens_out": tokens_out, "extra_cost_usd": cost}
