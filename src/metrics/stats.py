"""Statistics (EXPERIMENT_PLAN 12): cluster bootstrap, clustered logistic regression, McNemar + Holm."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap(units: pd.DataFrame, value: str = "diff", cluster: str = "table_id", B: int = 10000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Mean of `value` with a 95% CI from resampling clusters (tables) with replacement."""
    if units.empty:
        return (np.nan, np.nan, np.nan)
    g = units.groupby(cluster)[value].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    boots = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return float(s.sum() / c.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def vsr_with_ci(units: pd.DataFrame, by=("model", "support"), B: int = 10000) -> pd.DataFrame:
    rows = []
    for keys, g in units.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        m, lo, hi = cluster_bootstrap(g, B=B)
        rows.append({**dict(zip(by, keys)), "VSR": m, "ci_lo": lo, "ci_hi": hi, "n_units": len(g),
                     "p_h": g.p_h.mean(), "p_neutral": g.p_neutral.mean()})
    return pd.DataFrame(rows)


def logit_clustered(records: pd.DataFrame):
    """congruent ~ C(hyp)*C(support) + C(model), table-clustered SEs.  `records` in long form:
    one row per generation x evaluated hypothesis (neutral rows duplicated for pos/neg)."""
    import statsmodels.formula.api as smf
    d = records.dropna(subset=["c"]).copy()
    d["c"] = d["c"].astype(float)
    if d["c"].nunique() < 2:
        return None
    try:
        m = smf.glm("c ~ C(prompt, Treatment('neutral')) * C(support) + C(model)", data=d,
                    family=__import__("statsmodels.api").api.families.Binomial())
        return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["table_id"])[0]})
    except Exception as e:  # perfect separation etc.
        print("logit failed:", e)
        return None


def long_congruence(df: pd.DataFrame) -> pd.DataFrame:
    """Expand records so each generation is evaluated against h in {pos,neg}: c, prompt (hyp or neutral), support."""
    from src.vissyco.scenarios import support_level
    rows = []
    for h in ("pos", "neg"):
        sub = df[df.hyp.isin([h, "neutral"])].copy()
        sub["c"] = sub[f"cong_{h}"]
        sub["prompt"] = np.where(sub.hyp == "neutral", "neutral", "hyp")
        sub["support"] = [support_level(t, h) for t in sub.truth]
        sub["h"] = h
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def mcnemar_paired(a: pd.Series, b: pd.Series) -> dict:
    from statsmodels.stats.contingency_tables import mcnemar
    m = a.notna() & b.notna()
    a, b = a[m].astype(int), b[m].astype(int)
    t = [[int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())],
         [int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())]]
    r = mcnemar(t, exact=(t[0][1] + t[1][0]) < 25)
    return {"n": int(m.sum()), "b01": t[0][1], "b10": t[1][0], "stat": float(r.statistic), "p": float(r.pvalue)}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
