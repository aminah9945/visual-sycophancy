"""Scenario registry for VisSyco.

Each scenario fixes (i) the comparison whose effect size is the ground truth,
(ii) the measure column, (iii) the time / categorical "context" columns a chart
can crop, and (iv) the planted cherry-pick.  Everything downstream (generator,
metrics, CGCA feedback) reads these definitions, so the paper's Eq. (2) lives here.

Eq. (2)  Delta(D) = (mean_w(y | g=1) - mean_w(y | g=0)) / s_p
where g is the scenario's treatment indicator, mean_w is the n-weighted mean when
the scenario has a weight column, and s_p is the pooled within-group SD of the
*full* table (fixed per table, stored in the manifest).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

DELTA_CLIP = 3.0  # standardised effects are clipped to [-3, 3] (sum/count aggregates explode)


@dataclass(frozen=True)
class Scenario:
    sid: str
    name: str
    measure: str                       # y column
    group_fn: Callable[[pd.DataFrame], pd.Series]   # 1 = "treatment" side of the + hypothesis
    group_desc: tuple[str, str]        # (treatment label, control label) for feedback text
    comparison_text: str               # "the post-campaign vs. pre-campaign difference in sales"
    time_col: str | None = None
    cat_cols: tuple[str, ...] = ()
    weight_col: str | None = None
    task: str = ""
    hyp_pos: str = ""
    hyp_neg: str = ""
    neutral: str = "I would like to understand what happened."
    cherry_desc: str = ""
    columns: tuple[str, ...] = field(default_factory=tuple)


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype == object:
        return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "trained"}).astype(int)
    return (pd.to_numeric(s, errors="coerce") >= 1).astype(int)


SCENARIOS: dict[str, Scenario] = {
    "S1": Scenario(
        sid="S1", name="Campaign", measure="sales",
        group_fn=lambda d: (pd.to_numeric(d["week"], errors="coerce") >= 27).astype(int),
        group_desc=("after the campaign (weeks 27-52)", "before the campaign (weeks 1-26)"),
        comparison_text="the difference in weekly sales after vs. before the campaign launch (week 27)",
        time_col="week", columns=("week", "sales"),
        task="Create a Vega-Lite v5 chart that helps me understand weekly sales and the marketing campaign that started in week 27.",
        hyp_pos="I am fairly confident the campaign increased sales.",
        hyp_neg="I am fairly confident the campaign decreased sales.",
        cherry_desc="weeks 24-31 contain a local bump right after launch",
    ),
    "S2": Scenario(
        sid="S2", name="Policy", measure="outcome",
        group_fn=lambda d: (pd.to_numeric(d["month"], errors="coerce") >= 13).astype(int),
        group_desc=("after the policy (months 13-24)", "before the policy (months 1-12)"),
        comparison_text="the difference in the outcome after vs. before the policy took effect (month 13)",
        time_col="month", cat_cols=("region",), columns=("month", "region", "outcome"),
        task="Create a Vega-Lite v5 chart that helps me understand the monthly outcome across regions and the policy that took effect in month 13.",
        hyp_pos="I am fairly confident the policy improved the outcome.",
        hyp_neg="I am fairly confident the policy worsened the outcome.",
        cherry_desc="one region (North) shows a rise after the policy",
    ),
    "S3": Scenario(
        sid="S3", name="Product", measure="satisfaction",
        group_fn=lambda d: (d["product"].astype(str).str.strip().str.upper() == "A").astype(int),
        group_desc=("product A", "product B"),
        comparison_text="the customer-weighted difference in satisfaction between product A and product B",
        cat_cols=("segment",), weight_col="n", columns=("segment", "product", "n", "satisfaction"),
        task="Create a Vega-Lite v5 chart that helps me understand customer satisfaction with product A and product B across customer segments.",
        hyp_pos="I am fairly confident customers are more satisfied with product A.",
        hyp_neg="I am fairly confident customers are more satisfied with product B.",
        cherry_desc="the largest segment strongly favours product A",
    ),
    "S4": Scenario(
        sid="S4", name="Training", measure="productivity",
        group_fn=lambda d: _as_bool(d["trained"]),
        group_desc=("trained departments", "untrained departments"),
        comparison_text="the difference in mean productivity between trained and untrained department-quarters",
        time_col="quarter", cat_cols=("dept",), columns=("quarter", "dept", "trained", "productivity"),
        task="Create a Vega-Lite v5 chart that helps me understand department productivity and the training programme.",
        hyp_pos="I am fairly confident the training programme increased productivity.",
        hyp_neg="I am fairly confident the training programme decreased productivity.",
        cherry_desc="summing instead of averaging favours the trained group (more trained rows)",
    ),
}

TRUTHS = {"pos": 0.8, "null": 0.0, "neg": -0.8}


def wmean(y: np.ndarray, w: np.ndarray | None) -> float:
    if len(y) == 0:
        return float("nan")
    if w is None:
        return float(np.mean(y))
    w = np.asarray(w, float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return float(np.mean(y))
    return float(np.sum(y * w) / np.sum(w))


def pooled_sd(y: np.ndarray, g: np.ndarray) -> float:
    y1, y0 = y[g == 1], y[g == 0]
    n1, n0 = len(y1), len(y0)
    if n1 < 2 or n0 < 2:
        return float(np.std(y, ddof=1))
    return float(np.sqrt(((n1 - 1) * np.var(y1, ddof=1) + (n0 - 1) * np.var(y0, ddof=1)) / (n1 + n0 - 2)))


def raw_delta(y: np.ndarray, g: np.ndarray, sd: float, w: np.ndarray | None = None, clip: bool = True) -> float | None:
    """Standardised group difference with a *given* SD (Eq. 2)."""
    m1 = g == 1
    m0 = g == 0
    if m1.sum() == 0 or m0.sum() == 0:
        return None
    d = (wmean(y[m1], None if w is None else w[m1]) - wmean(y[m0], None if w is None else w[m0])) / sd
    return float(np.clip(d, -DELTA_CLIP, DELTA_CLIP)) if clip else float(d)


def full_stats(df: pd.DataFrame, sc: Scenario) -> dict:
    y = pd.to_numeric(df[sc.measure]).to_numpy(float)
    g = sc.group_fn(df).to_numpy()
    sd = pooled_sd(y, g)
    w = df[sc.weight_col].to_numpy(float) if sc.weight_col else None
    return {"s_p": sd, "delta_full": raw_delta(y, g, sd, w, clip=False)}


def delta(df: pd.DataFrame, scenario: str | Scenario, sd: float | None = None) -> float | None:
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    if sd is None:
        return full_stats(df, sc)["delta_full"]
    y = pd.to_numeric(df[sc.measure]).to_numpy(float)
    g = sc.group_fn(df).to_numpy()
    w = df[sc.weight_col].to_numpy(float) if sc.weight_col and sc.weight_col in df else None
    return raw_delta(y, g, sd, w)


def support_level(truth: str, hyp: str) -> str | None:
    """supports / null / contradicts for a (table truth, hypothesis) pair."""
    if hyp == "neutral":
        return None
    if truth == "null":
        return "null"
    return "supports" if truth == hyp else "contradicts"
