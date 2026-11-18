"""Write LaTeX tables / PDF figures into paper/tables and paper/figures.

    python -m src.analysis.make_tables --exp DATA   # VisSyco statistics table + scenario figure
    python -m src.analysis.make_tables --exp A      # ChartAttack reported vs reproduced
    python -m src.analysis.make_tables --exp E1     # CSR table + figure + regression
    python -m src.analysis.make_tables --exp E2     # defence comparison + McNemar
    python -m src.analysis.make_tables --exp ABL    # ablations + tau curve
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.grounded import load_table, manifest, vsr_units
from src.metrics.stats import holm, logit_clustered, long_congruence, mcnemar_paired, vsr_with_ci
from src.vissyco.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
RUNS = ROOT / "runs"
SUPPORT_ORDER = ["supports", "null", "contradicts"]
MODEL_NAMES = {"qwen3-30b": "Qwen3-30B-A3B", "gpt-4o-mini": "GPT-4o-mini", "claude-haiku-5.5": "Claude Haiku 5.5",
               "qwen3-vl-2b": "Qwen3-VL-2B", "internvl3.5-2b": "InternVL-3.5-2B", "qwen3-vl-4b": "Qwen3-VL-4B"}
DEF_NAMES = {"B0": "No defence", "B1": "Prompt guard", "B2": "Self-Refine", "B3": "Linter", "CGCA": "\\textbf{CGCA (ours)}"}


def f(x, d=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{100 * x:.1f}" if pct else f"{x:.{d}f}"


def write(name, s):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(s, encoding="utf-8")
    print("wrote", TAB / f"{name}.tex")


def load(exp):
    p = RUNS / exp / "summary.csv"
    return pd.read_csv(p) if p.exists() else None


# ----------------------------------------------------------------------------- DATA
def data_tables():
    m = pd.DataFrame(manifest().values())
    rows = []
    for (sid, truth), g in m.groupby(["scenario", "truth"], sort=False):
        rows.append((sid, truth, int(g.n_rows.iloc[0]), len(g["columns"].iloc[0]), g.delta_full.mean(), g.delta_full.min(),
                     g.delta_full.max(), g.delta_cherry.mean()))
    order = {"pos": 0, "null": 1, "neg": 2}
    rows.sort(key=lambda r: (r[0], order[r[1]]))
    lab = {"pos": "supports $h^+$", "null": "null", "neg": "contradicts $h^+$"}
    body = "\n".join(f"{SCENARIOS[s].sid} {SCENARIOS[s].name} & {lab[t]} & {n} & {c} & {mu:+.2f} [{lo:+.2f}, {hi:+.2f}] & {dc:+.2f} \\\\"
                     for s, t, n, c, mu, lo, hi, dc in rows)
    write("vissyco_stats", r"""\begin{table}[t]
\centering
\caption{VisSyco statistics (3 seeds per cell; seed 1 = dev, seeds 2--3 = test). $\Delta(D)$: full-table effect, mean [min, max]; $\Delta(D_{W^\ast})$: effect on the planted cherry subset (clipped at 3).}
\label{tab:vissyco}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
Scenario & Condition & Rows & Cols & $\Delta(D)$ & $\Delta(D_{W^\ast})$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # sample rows table
    smp = []
    for sid in SCENARIOS:
        d = load_table(f"{sid}_null_s2")
        smp.append(f"{sid} & " + ", ".join(d.columns) + " & " + "; ".join(
            "(" + ", ".join(str(round(v, 2)) if isinstance(v, float) else str(v) for v in r) + ")" for r in d.head(2).itertuples(index=False)) + " \\\\")
    write("vissyco_sample", r"""\begin{table}[t]
\centering
\caption{Sample rows (null condition, seed 2).}
\label{tab:vissyco-sample}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lp{2.6cm}p{4.6cm}}
\toprule
 & Columns & First rows \\
\midrule
""" + "\n".join(smp) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # figure: one table per scenario (contradicts condition) with the cherry region shaded
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.3))
    d = load_table("S1_neg_s2")
    ax[0].plot(d.week, d.sales, lw=1, color="#3b6ea8")
    ax[0].axvspan(24, 31, color="orange", alpha=.25)
    ax[0].axvline(27, ls="--", c="k", lw=.7)
    ax[0].set(title="S1 Campaign (window)", xlabel="week", ylabel="sales")
    d = load_table("S2_neg_s2")
    for r, g in d.groupby("region"):
        ax[1].plot(g.month, g.outcome, lw=1.6 if r == "North" else .8, alpha=1 if r == "North" else .5, label=r)
    ax[1].axvline(13, ls="--", c="k", lw=.7)
    ax[1].legend(fontsize=5, frameon=False)
    ax[1].set(title="S2 Policy (subgroup)", xlabel="month", ylabel="outcome")
    d = load_table("S3_neg_s2")
    p = d.pivot(index="segment", columns="product", values="satisfaction")
    xs = np.arange(len(p))
    ax[2].bar(xs - .2, p["A"], .4, label="A")
    ax[2].bar(xs + .2, p["B"], .4, label="B")
    ci = list(p.index).index("Professionals")
    ax[2].axvspan(ci - .5, ci + .5, color="orange", alpha=.25)
    ax[2].set_xticks(xs, [s[:4] for s in p.index], fontsize=6)
    ax[2].set_ylim(p.values.min() - 1, p.values.max() + .5)
    ax[2].legend(fontsize=6, frameon=False)
    ax[2].set(title="S3 Product (segment)", ylabel="satisfaction")
    d = load_table("S4_neg_s2")
    g = d.groupby("trained").productivity
    ax[3].bar([0, 1], g.mean(), .35, label="mean")
    ax3 = ax[3].twinx()
    ax3.bar([0.4, 1.4], g.sum(), .35, color="orange", alpha=.6, label="sum")
    ax[3].set_xticks([0.2, 1.2], ["untrained", "trained"])
    ax[3].set(title="S4 Training (aggregation)", ylabel="mean (blue)")
    ax3.set_ylabel("sum (orange)")
    for a in ax:
        a.tick_params(labelsize=6)
        a.title.set_size(8)
        a.xaxis.label.set_size(7)
        a.yaxis.label.set_size(7)
    fig.tight_layout()
    fig.savefig(FIG / "vissyco_scenarios.pdf")
    fig.savefig(FIG / "vissyco_scenarios.png", dpi=200)
    print("wrote", FIG / "vissyco_scenarios.pdf")


# ----------------------------------------------------------------------------- A
def phase_a():
    r = pd.read_csv(RUNS / "A" / "results.csv")
    rows = []
    for _, x in r.iterrows():
        rows.append(f"{MODEL_NAMES.get(x.model, x.model)} & {f(x.acc_correct, pct=True)} & {f(x.acc_misleading, pct=True)} & "
                    f"{x.drop_pp:.1f} & {f(x.DR_orig_correct, pct=True)} & {f(x.DR_orig_incorrect, pct=True)} \\\\")
    write("phaseA", r"""\begin{table}[t]
\centering
\caption{ChartAttack Tier-1 reproduction on AttackViz (PlotQA in-domain test, $N=500$ questions stratified by chart type; each question is answered on its correct chart and on all its misleading variants). RA: relaxed accuracy (\%); DR: deception rate (\%), originally correct / incorrect.}
\label{tab:repro}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
Reader & RA correct & RA misleading & Drop (pp) & DR\textsubscript{corr} & DR\textsubscript{inc} \\
\midrule
\multicolumn{6}{l}{\emph{Reported~\cite{ortizbarajas2026chartattack} (all 19 readers)}} \\
Average & -- & -- & 17.2 & 11.2 & 1.7 \\
Range & -- & -- & 4.4--26.6 & -- & -- \\
\midrule
\multicolumn{6}{l}{\emph{Reproduced (ours)}} \\
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ----------------------------------------------------------------------------- E1
def e1(exp="E1"):
    df = load(exp)
    if df is None:
        print("no", exp)
        return
    units = vsr_units(df)
    ci = vsr_with_ci(units)
    ci.to_csv(RUNS / exp / "vsr_ci.csv", index=False)
    models = list(dict.fromkeys(df.model))
    g = df.groupby("model")
    lines = []
    for m in models:
        d = df[df.model == m]
        c = ci[ci.model == m].set_index("support")
        hyp = d[d.hyp != "neutral"]
        ei = hyp.EI.astype(float)
        cells = []
        for s in SUPPORT_ORDER:
            if s in c.index:
                x = c.loc[s]
                cells.append(f"{x.VSR:+.2f} [{x.ci_lo:+.2f}, {x.ci_hi:+.2f}]")
            else:
                cells.append("--")
        lines.append(f"{MODEL_NAMES.get(m, m)} & " + " & ".join(cells) + f" & {f(ei.mean())} & {f(d.valid.mean(), pct=True)} & {f(d.fid.astype(float).mean(), pct=True)} \\\\")
    write("e1_vsr", r"""\begin{table*}[t]
\centering
\caption{E1: Chart Sycophancy Rate (Eq.~\ref{eq:vsr}) by support level, 95\% cluster-bootstrap CIs over tables (10{,}000 resamples). Mean EI over hypothesis prompts (Eq.~\ref{eq:ei}); Val: valid specs (\%); Fid: task fidelity (\%).}
\label{tab:e1}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Generator & CSR supports & CSR null & CSR contradicts & Mean EI & Val & Fid \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    # deviation breakdown
    dv = []
    for m in models:
        d = df[(df.model == m) & df.valid.astype(bool)]
        cong = d[d.congruent == 1]
        row = [MODEL_NAMES.get(m, m), f(d.congruent.astype(float).mean(), pct=True) if d.congruent.notna().any() else "--"]
        for k in ("window", "subgroup", "aggregation", "encoding"):
            row.append(f(d[f"dev_{k}"].astype(float).mean(), pct=True))
        row.append(f((d.ei_status == "unsupported").mean(), pct=True))
        row.append(f(d.ei_status.isin(["no_comparison"]).mean(), pct=True))
        dv.append(" & ".join(row) + " \\\\")
    write("e1_dev", r"""\begin{table}[t]
\centering
\caption{E1: share of valid specs (\%) that are congruent with the prompt's hypothesis, by deviation type (all prompts), unsupported by the executor, and with no visible comparison.}
\label{tab:e1dev}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcccccccc}
\toprule
Generator & Cong. & Window & Subgr. & Aggr. & Enc. & Unsupp. & No comp. \\
\midrule
""" + "\n".join(dv) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # regression
    res = logit_clustered(long_congruence(df))
    if res is not None:
        (RUNS / exp / "logit.txt").write_text(str(res.summary()))
        tab = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues})
        tab.to_csv(RUNS / exp / "logit.csv")
    # figure
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        c = ci[ci.model == m].set_index("support").reindex(SUPPORT_ORDER)
        xs = np.arange(3) + (i - (len(models) - 1) / 2) * w
        ax.bar(xs, c.VSR, w, label=MODEL_NAMES.get(m, m), yerr=[c.VSR - c.ci_lo, c.ci_hi - c.VSR], capsize=2, error_kw={"lw": .7})
    ax.axhline(0, c="k", lw=.6)
    ax.set_xticks(range(3), SUPPORT_ORDER)
    ax.set_ylabel("CSR")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "e1_vsr.pdf")
    fig.savefig(FIG / "e1_vsr.png", dpi=200)


# ----------------------------------------------------------------------------- E2 / ABL
def defence_metrics(df: pd.DataFrame, defs: list[str]) -> pd.DataFrame:
    rows = []
    for m in dict.fromkeys(df.model):
        for d in defs:
            x = df[(df.model == m) & (df.defense == d)]
            if x.empty:
                continue
            u = vsr_units(x)
            neu = x[(x.hyp == "neutral") & x.valid.astype(bool)]
            frr = neu["flagged_r0"].astype(float).mean() if "flagged_r0" in neu and neu["flagged_r0"].notna().any() else \
                (neu["revised"].astype(float).mean() if "revised" in neu and d in ("B2", "B3") else 0.0)
            calls = 1 + x["extra_calls"].fillna(0).astype(float).mean() if "extra_calls" in x else 1
            tin = x.tokens_in.fillna(0) + (x["extra_tokens_in"].fillna(0) if "extra_tokens_in" in x else 0)
            tout = x.tokens_out.fillna(0) + (x["extra_tokens_out"].fillna(0) if "extra_tokens_out" in x else 0)
            hyp = x[x.hyp != "neutral"]
            rows.append({"model": m, "defense": d, "VSR": u["diff"].mean() if len(u) else np.nan,
                         "VSR_contra": u[u.support == "contradicts"]["diff"].mean() if len(u) else np.nan,
                         "cong": hyp.congruent.astype(float).mean(), "EI": hyp.EI.astype(float).mean(), "FRR": frr,
                         "Val": x.valid.astype(float).mean(), "Fid": x.fid.astype(float).mean(),
                         "calls": calls, "ktok": (tin + tout).mean() / 1000})
    return pd.DataFrame(rows)


def mcnemar_vs(df, ref="CGCA", others=("B0", "B1", "B2", "B3")):
    out = []
    key = ["model", "table_id", "hyp", "sample"]
    h = df[df.hyp != "neutral"]
    r = h[h.defense == ref].set_index(key).congruent
    for o in others:
        x = h[h.defense == o].set_index(key).congruent
        j = pd.concat([r.rename("ref"), x.rename("o")], axis=1, join="inner")
        if len(j):
            out.append({"vs": o, **mcnemar_paired(j["o"], j["ref"])})
    if out:
        adj = holm([o["p"] for o in out])
        for o, p in zip(out, adj):
            o["p_holm"] = p
    return pd.DataFrame(out)


def e2(exp="E2"):
    df = load(exp)
    if df is None:
        print("no", exp)
        return
    defs = [d for d in ("B0", "B1", "B2", "B3", "CGCA") if d in set(df.defense)]
    t = defence_metrics(df, defs)
    t.to_csv(RUNS / exp / "defences.csv", index=False)
    mc = mcnemar_vs(df)
    mc.to_csv(RUNS / exp / "mcnemar.csv", index=False)
    lines = []
    for m in dict.fromkeys(t.model):
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{MODEL_NAMES.get(m, m)}}}}} \\\\")
        for _, x in t[t.model == m].iterrows():
            lines.append(f"\\quad {DEF_NAMES.get(x.defense, x.defense)} & {x.VSR:+.2f} & {x.VSR_contra:+.2f} & {f(x.cong, pct=True)} & {f(x.EI)} & "
                         f"{f(x.FRR, pct=True)} & {f(x.Val, pct=True)} & {f(x.Fid, pct=True)} & {x.calls:.2f} / {x.ktok:.1f} \\\\")
    write("e2_main", r"""\begin{table*}[t]
\centering
\caption{E2: defences in the benign setting (test split). CSR overall and on contradicting data; Cong.: congruent-deviation rate on hypothesis prompts (\%); EI: mean effect inflation; FRR: false revision rate on neutral prompts (\%); Val, Fid (\%); cost: generator calls / k tokens per request. Post-hoc defences (B2, B3, CGCA) start from the same E1 generation.}
\label{tab:e2}
\footnotesize
\begin{tabular}{lcccccccc}
\toprule
Defence & CSR & CSR\textsubscript{contra} & Cong. & EI & FRR & Val & Fid & Cost \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    if len(mc):
        write("e2_mcnemar", r"""\begin{table}[t]
\centering
\caption{McNemar tests of CGCA vs.\ each baseline on paired congruent outcomes (hypothesis prompts, all generators). $b_{01}$: baseline congruent, CGCA not; $b_{10}$: the reverse. Holm-adjusted $p$.}
\label{tab:mcnemar}
\footnotesize
\begin{tabular}{lcccc}
\toprule
vs. & $n$ & $b_{01}$ & $b_{10}$ & $p_\text{Holm}$ \\
\midrule
""" + "\n".join(f"{DEF_NAMES.get(r.vs, r.vs)} & {r.n} & {r.b01} & {r.b10} & {r.p_holm:.3g} \\\\" for r in mc.itertuples()) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def abl(exp="ABL"):
    df = load(exp)
    if df is None:
        print("no", exp)
        return
    defs = list(dict.fromkeys(["B0", "CGCA"] + sorted(d for d in set(df.defense) if d.startswith("A"))))
    t = defence_metrics(df, defs)
    t.to_csv(RUNS / exp / "ablations.csv", index=False)
    agg = t.groupby("defense", sort=False)[["VSR", "VSR_contra", "cong", "EI", "FRR", "Val", "Fid", "calls"]].mean().reindex(defs)
    label = {"B0": "No defence", "CGCA": "CGCA (EI+flags, $R=2$, grounded, aug.)", "A1-EIonly": "A1: EI only", "A1-flagsonly": "A1: flags only",
             "A2-R0": "A2: $R=0$ (augment only)", "A2-R1": "A2: $R=1$", "A3-generic": "A3: generic feedback",
             "A4-agnostic": "A4: hypothesis-agnostic $|\\EI|$", "A5-noaug": "A5: no augmentation"}
    lines = [f"{label.get(d, d)} & {x.VSR:+.2f} & {x.VSR_contra:+.2f} & {f(x.cong, pct=True)} & {f(x.EI)} & {f(x.FRR, pct=True)} & {f(x.Fid, pct=True)} & {x.calls:.2f} \\\\"
             for d, x in agg.iterrows() if not d.startswith("A6")]
    write("ablations", r"""\begin{table}[t]
\centering
\caption{Ablations (test split, averaged over the ablation generators).}
\label{tab:abl}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccccccc}
\toprule
Variant & CSR & CSR\textsubscript{c} & Cong. & EI & FRR & Fid & Calls \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    cur = agg[[d.startswith("A6") or d == "CGCA" for d in agg.index]].copy()
    cfg_tau = df[df.defense == "CGCA"].tau_ei.dropna()
    cur["tau"] = [float(d.split("tau")[1]) if d.startswith("A6") else (cfg_tau.iloc[0] if len(cfg_tau) else np.nan) for d in cur.index]
    cur = cur.sort_values("tau")
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(cur.FRR, cur.VSR, "o-")
    for _, x in cur.iterrows():
        ax.annotate(f"{x.tau:g}", (x.FRR, x.VSR), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="FRR (neutral prompts)", ylabel="CSR")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "abl_tau.pdf")
    fig.savefig(FIG / "abl_tau.png", dpi=200)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E1")
    a = ap.parse_args(argv)
    {"DATA": data_tables, "A": phase_a, "E1": e1, "E2": e2, "ABL": abl}[a.exp]()


if __name__ == "__main__":
    main()
