"""Phase E: visual-analytics auditing interface (paper Sec. 3.3, views V1-V4).

    streamlit run dashboard/app.py

Reads runs/<exp>/raw/calls.jsonl (E1, E2, ABL ...).  Coordinated views: a click in V1 filters
V2-V4; a click in V2 filters V3.  Works fully offline on cached results (for live demos).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cgca.audit import _overview  # noqa: E402
from src.metrics.grounded import manifest, vsr_units  # noqa: E402

st.set_page_config(page_title="Visual Sycophancy Auditor", layout="wide")
DEV_TYPES = ["window", "subgroup", "aggregation", "encoding"]


@st.cache_data
def load(exp: str) -> pd.DataFrame:
    recs = [json.loads(l) for l in open(ROOT / "runs" / exp / "raw" / "calls.jsonl", encoding="utf-8")]
    df = pd.DataFrame(recs)
    if "defense" not in df:
        df["defense"] = "none"
    df["defense"] = df["defense"].replace({"none": "B0"})
    for k in DEV_TYPES:
        df[f"dev_{k}"] = df["dev"].map(lambda d: (d or {}).get(k, 0))
    df["rid"] = range(len(df))
    return df


exps = sorted(p.parent.parent.name for p in (ROOT / "runs").glob("*/raw/calls.jsonl"))
st.sidebar.title("Visual Sycophancy Auditor")
if not exps:
    st.info("No results yet. Run `python -m src.run_generation --exp E1` first.")
    st.stop()
exp = st.sidebar.selectbox("Experiment", exps, index=exps.index("E2") if "E2" in exps else 0)
df = load(exp)
defs = sorted(df.defense.unique())
defense = st.sidebar.selectbox("Defence", defs, index=defs.index("B0") if "B0" in defs else 0)
scen = st.sidebar.multiselect("Scenarios", sorted(df.scenario.unique()), default=sorted(df.scenario.unique()))
d = df[(df.defense == defense) & df.scenario.isin(scen)]
st.sidebar.metric("Generations", len(d))
st.sidebar.metric("Valid specs", f"{d.valid.mean():.0%}")

# ------------------------------------------------------------------ V1 overview
st.subheader("V1 · Overview: Visual Sycophancy Rate by model × hypothesis × support")
u = vsr_units(d)
if u.empty:
    st.warning("No VSR units for this selection.")
    st.stop()
u["cell"] = u.h.map({"pos": "h+", "neg": "h−"}) + " · " + u.support
v1 = u.groupby(["model", "cell", "h", "support"], as_index=False).agg(VSR=("diff", "mean"), p_h=("p_h", "mean"),
                                                                     p_neutral=("p_neutral", "mean"), n=("diff", "size"))
sel1 = alt.selection_point(fields=["model", "h", "support"], name="v1")
lim = max(0.05, float(v1.VSR.abs().max()))
heat = (alt.Chart(v1).mark_rect(stroke="white").encode(
    x=alt.X("cell:N", title=None, sort=[f"{h} · {s}" for h in ("h+", "h−") for s in ("supports", "null", "contradicts")]),
    y=alt.Y("model:N", title=None),
    color=alt.Color("VSR:Q", scale=alt.Scale(scheme="redblue", reverse=True, domain=[-lim, lim]), title="VSR"),
    opacity=alt.condition(sel1, alt.value(1), alt.value(0.35)),
    tooltip=["model", "cell", alt.Tooltip("VSR:Q", format="+.2f"), alt.Tooltip("p_h:Q", format=".2f"),
             alt.Tooltip("p_neutral:Q", format=".2f"), "n"])
    .add_params(sel1).properties(height=40 * v1.model.nunique() + 40))
text = alt.Chart(v1).mark_text(fontSize=11).encode(x=alt.X("cell:N", sort=None), y="model:N", text=alt.Text("VSR:Q", format="+.2f"))
ev1 = st.altair_chart(heat + text, use_container_width=True, on_select="rerun", key="v1")
picked = (ev1.selection.get("v1") or []) if ev1 else []
focus = d
if picked:
    p = picked[0]
    hyp_ok = d.hyp.isin([p["h"], "neutral"])
    tids = u[(u.model == p["model"]) & (u.h == p["h"]) & (u.support == p["support"])].table_id
    focus = d[(d.model == p["model"]) & hyp_ok & d.table_id.isin(tids)]
    st.caption(f"Filtered to **{p['model']}**, hypothesis **{p['h']}**, support **{p['support']}** ({len(focus)} generations).")

# ------------------------------------------------------------------ V2 attribution
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("V2 · Attribution")
    hyp_rows = focus[focus.hyp != "neutral"]
    cong = hyp_rows[hyp_rows.congruent == 1]
    att = pd.DataFrame([{"model": m, "type": k, "count": int(g[f"dev_{k}"].sum())}
                        for m, g in cong.groupby("model") for k in DEV_TYPES])
    sel2 = alt.selection_point(fields=["type"], name="v2")
    if att.empty:
        st.write("No hypothesis-congruent charts in this selection.")
        ev2 = None
    else:
        bars = (alt.Chart(att).mark_bar().encode(
            y=alt.Y("model:N", title=None), x=alt.X("count:Q", stack="zero", title="congruent charts with deviation"),
            color=alt.Color("type:N", scale=alt.Scale(domain=DEV_TYPES)), opacity=alt.condition(sel2, alt.value(1), alt.value(.35)),
            tooltip=["model", "type", "count"]).add_params(sel2).properties(height=200))
        ev2 = st.altair_chart(bars, use_container_width=True, on_select="rerun", key="v2")
    picked2 = (ev2.selection.get("v2") or []) if ev2 else []
    insp = hyp_rows[hyp_rows.congruent == 1] if len(hyp_rows[hyp_rows.congruent == 1]) else focus
    if picked2:
        insp = insp[insp[f"dev_{picked2[0]['type']}"] == 1]
        st.caption(f"Inspector shows **{picked2[0]['type']}** deviations.")
    only_cong = st.checkbox("Inspector: only congruent charts", value=True)
    if not only_cong:
        insp = focus

# ------------------------------------------------------------------ V3 inspector
with c2:
    st.subheader("V3 · Inspector: generated chart vs. full-data context")
    insp = insp.sort_values(["model", "table_id", "hyp", "sample"])
    if insp.empty:
        st.write("Nothing to inspect.")
    else:
        i = st.number_input(f"Sample (1-{len(insp)})", 1, len(insp), 1) - 1
        r = insp.iloc[int(i)]
        meta = manifest()[r.table_id]
        st.markdown(f"**{r.model}** · `{r.table_id}` · prompt **{r.hyp}** · support **{r.support}** · defence **{r.defense}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Δ(D) full", f"{r.delta_full:+.2f}")
        m2.metric("Δ(D_S) plotted", "—" if pd.isna(r.delta_plot) else f"{r.delta_plot:+.2f}")
        m3.metric("EI (prompt h)", "—" if pd.isna(r.EI) else f"{r.EI:+.2f}")
        m4.metric("LF", "—" if pd.isna(r.LF) else f"{r.LF:.2f}")
        st.caption(f"Selection: {r.selection_text or '—'} · flags: {', '.join(k for k, v in (r.flags or {}).items() if v) or 'none'}")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("*Generated chart*")
            if isinstance(r.spec, dict) and r.valid:
                try:
                    st.vega_lite_chart(r.spec, use_container_width=True)
                except Exception as e:
                    st.code(f"render error: {e}")
            else:
                st.code((r.response or "")[:1500])
        with g2:
            st.markdown("*Full data, selected region highlighted*")
            ov = _overview(r.table_id, r.selection if isinstance(r.selection, dict) else {})
            ov["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
            st.vega_lite_chart(ov, use_container_width=True)
        with st.expander("Prompt response / spec JSON"):
            st.json(r.spec if isinstance(r.spec, dict) else {})

# ------------------------------------------------------------------ V4 revisions
st.subheader("V4 · CGCA revision trajectories (EI across rounds)")
cg = df[(df.defense == "CGCA") & df.scenario.isin(scen)] if "CGCA" in set(df.defense) else pd.DataFrame()
if picked and len(cg):
    cg = cg[(cg.model == picked[0]["model"]) & cg.table_id.isin(focus.table_id)]
if cg.empty or "trajectory" not in cg:
    st.write("Run E2 with CGCA to see revision trajectories.")
else:
    rows = []
    for _, r in cg[cg.revised == True].iterrows():
        fbs = r.feedbacks if isinstance(r.feedbacks, list) else []
        for k, e in enumerate(r.trajectory or []):
            rows.append({"req": f"{r.model}|{r.table_id}|{r.hyp}|{r['sample']}", "round": k, "EI": e,
                         "outcome": "augmented" if r.augmented else "fixed by revision",
                         "feedback": (fbs[k] if k < len(fbs) else "accepted / final")[:400]})
    tr = pd.DataFrame(rows).dropna(subset=["EI"])
    if tr.empty:
        st.write("No revised requests in this selection.")
    else:
        hover = alt.selection_point(fields=["req"], on="pointerover", name="hover")
        ln = (alt.Chart(tr).mark_line(point=True).encode(
            x=alt.X("round:O"), y=alt.Y("EI:Q"), detail="req:N", color=alt.Color("outcome:N"),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.25)), tooltip=["req", "round", alt.Tooltip("EI:Q", format="+.2f"), "feedback"])
            .add_params(hover).properties(height=260))
        st.altair_chart(ln, use_container_width=True)
        st.caption(f"{tr.req.nunique()} revised requests · {cg.augmented.mean():.0%} of CGCA requests ended in augmentation.")
