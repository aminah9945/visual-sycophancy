"""Phase B3: Vega-Lite v5 spec executor (pandas).

Validates a spec against the official v5 JSON schema and re-executes its data
pipeline so we know exactly which rows/aggregates the chart *plots*:

  inline data.values (or top-level datasets) -> view transforms in order
  (filter, timeUnit, aggregate, joinaggregate, window[rank/cumulative], calculate,
  fold) -> encoding-level timeUnit/aggregate -> x-axis scale.domain clipping.

Every plotted row carries ``__src``: the tuple of inline row ids it was computed
from, so metrics can recover which part of the data a chart shows.

Expressions (filter / calculate strings) are parsed with a whitelisted AST
evaluator - LLM-written code is never passed to ``eval``.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_PATH = Path(__file__).with_name("vega-lite-v5.json")
POS_CHANNELS = ("x", "y", "x2", "y2", "theta", "radius")
GROUP_CHANNELS = ("x", "y", "x2", "y2", "color", "fill", "stroke", "shape", "size", "opacity",
                  "detail", "column", "row", "facet", "xOffset", "yOffset", "theta", "radius", "text", "strokeDash")
AGG_OPS = {
    "mean": "mean", "average": "mean", "sum": "sum", "median": "median", "min": "min", "max": "max",
    "count": "count", "valid": "count", "distinct": "nunique", "stdev": "std", "variance": "var",
    "q1": lambda s: s.quantile(0.25), "q3": lambda s: s.quantile(0.75),
}


class Unsupported(Exception):
    pass


# ----------------------------------------------------------------------------- validation
@lru_cache(maxsize=1)
def _validator():
    import jsonschema
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema)


def validate(spec: Any) -> list[str]:
    if not isinstance(spec, dict):
        return ["spec is not a JSON object"]
    errs = []
    for e in _validator().iter_errors(spec):
        errs.append(f"{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}")
        if len(errs) >= 5:
            break
    return errs


# ----------------------------------------------------------------------------- expressions
_JS_SUBS = [
    (r"===", "=="), (r"!==", "!="), (r"&&", " and "), (r"\|\|", " or "),
    (r"!(?!=)", " not "), (r"\btrue\b", "True"), (r"\bfalse\b", "False"), (r"\bnull\b", "None"),
]


def js_to_py(expr: str) -> str:
    if "?" in expr or "=>" in expr:
        raise Unsupported(f"expression not supported: {expr}")
    e = re.sub(r"datum\[\s*['\"]([^'\"]+)['\"]\s*\]", lambda m: f"datum[{m.group(1)!r}]", expr)
    e = re.sub(r"datum\.([A-Za-z_]\w*)", lambda m: f"datum[{m.group(1)!r}]", e)
    for a, b in _JS_SUBS:
        e = re.sub(a, b, e)
    return e


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _dt(v):
    return pd.to_datetime(v, errors="coerce")


EXPR_FUNCS = {
    "toNumber": lambda v: _num(v), "toString": str, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "pow": pow,
    "min": min, "max": max, "isValid": lambda v: v is not None and not (isinstance(v, float) and math.isnan(v)),
    "year": lambda v: _dt(v).year, "month": lambda v: _dt(v).month - 1, "date": lambda v: _dt(v).day,
    "quarter": lambda v: (_dt(v).month - 1) // 3 + 1, "toDate": _dt, "datetime": lambda *a: pd.Timestamp(*a),
    "indexof": lambda a, b: a.find(b) if isinstance(a, str) else -1,
    "parseInt": lambda v: int(float(v)), "parseFloat": float,
    "substring": lambda s, a, b=None: str(s)[a:b], "upper": lambda s: str(s).upper(), "lower": lambda s: str(s).lower(),
    "test": lambda r, s: bool(re.search(r, str(s))) if isinstance(r, str) else False,
}
_ALLOWED = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load,
            ast.Constant, ast.Subscript, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.In, ast.NotIn, ast.Index if hasattr(ast, "Index") else ast.Constant, ast.List, ast.Tuple,
            ast.Attribute)


@lru_cache(maxsize=4096)
def compile_expr(expr: str):
    py = js_to_py(expr)
    try:
        tree = ast.parse(py.strip(), mode="eval")
    except SyntaxError as e:
        raise Unsupported(f"unparseable expression: {expr}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise Unsupported(f"expression node {type(node).__name__} in {expr}")
        if isinstance(node, ast.Attribute) and node.attr not in {"length"}:
            raise Unsupported(f"attribute access in {expr}")
        if isinstance(node, ast.Name) and node.id not in EXPR_FUNCS and node.id not in {"datum", "True", "False", "None", "PI", "E"}:
            raise Unsupported(f"unknown name {node.id} in {expr}")
    return tree


def _ev(node, datum):
    t = type(node)
    if t is ast.Expression:
        return _ev(node.body, datum)
    if t is ast.Constant:
        return node.value
    if t is ast.Name:
        return {"datum": datum, "True": True, "False": False, "None": None, "PI": math.pi, "E": math.e}.get(node.id, EXPR_FUNCS.get(node.id))
    if t is ast.Subscript:
        key = _ev(node.slice, datum)
        base = _ev(node.value, datum)
        return base.get(key) if isinstance(base, dict) else base[key]
    if t is ast.Attribute:
        return len(_ev(node.value, datum))
    if t is ast.BoolOp:
        vals = (_ev(v, datum) for v in node.values)
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if t is ast.UnaryOp:
        v = _ev(node.operand, datum)
        return (not v) if isinstance(node.op, ast.Not) else (-_num(v) if isinstance(node.op, ast.USub) else _num(v))
    if t is ast.BinOp:
        a, b = _ev(node.left, datum), _ev(node.right, datum)
        if isinstance(node.op, ast.Add) and (isinstance(a, str) or isinstance(b, str)):
            return f"{a}{b}"
        a, b = _num(a), _num(b)
        op = node.op
        return (a + b if isinstance(op, ast.Add) else a - b if isinstance(op, ast.Sub) else a * b if isinstance(op, ast.Mult)
                else (a / b if b else float("nan")) if isinstance(op, ast.Div) else a % b if isinstance(op, ast.Mod) else a ** b)
    if t is ast.Compare:
        left = _ev(node.left, datum)
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, datum)
            l2, r2 = (_num(left), _num(right)) if not isinstance(op, (ast.Eq, ast.NotEq)) else (left, right)
            if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(left, (int, float)) != isinstance(right, (int, float)):
                l2, r2 = str(left), str(right)  # JS loose equality on mixed types
            try:
                ok = {ast.Eq: l2 == r2, ast.NotEq: l2 != r2}.get(type(op))
                if ok is None:
                    ok = {ast.Lt: lambda: l2 < r2, ast.LtE: lambda: l2 <= r2, ast.Gt: lambda: l2 > r2,
                          ast.GtE: lambda: l2 >= r2, ast.In: lambda: l2 in r2, ast.NotIn: lambda: l2 not in r2}[type(op)]()
            except TypeError:
                ok = False
            if not ok:
                return False
            left = right
        return True
    if t is ast.Call:
        f = _ev(node.func, datum)
        return f(*[_ev(a, datum) for a in node.args])
    if t in (ast.List, ast.Tuple):
        return [_ev(e, datum) for e in node.elts]
    raise Unsupported(f"node {t.__name__}")


def eval_expr(expr: str, df: pd.DataFrame) -> pd.Series:
    tree = compile_expr(expr)
    recs = df.drop(columns=["__src"], errors="ignore").to_dict("records")
    out = []
    for r in recs:
        try:
            out.append(_ev(tree, r))
        except Unsupported:
            raise
        except Exception:
            out.append(None)
    return pd.Series(out, index=df.index, dtype=object)


# ----------------------------------------------------------------------------- time units
def apply_timeunit(s: pd.Series, unit: str) -> pd.Series:
    unit = unit.get("unit", "") if isinstance(unit, dict) else unit
    unit = unit.replace("utc", "")
    if pd.api.types.is_numeric_dtype(s) and not unit.startswith("year"):
        return s  # e.g. integer weeks - nothing to truncate
    d = pd.to_datetime(s, errors="coerce")
    if d.isna().all():
        return s
    if unit == "year":
        return d.dt.year
    if unit == "month":
        return d.dt.month
    if unit == "quarter":
        return d.dt.quarter
    if unit in ("yearmonth", "yearmonthdate", "yearquarter", "yearweek", "date", "week"):
        freq = {"yearmonth": "M", "yearquarter": "Q", "yearweek": "W", "week": "W"}.get(unit, "D")
        return d.dt.to_period(freq).dt.start_time
    raise Unsupported(f"timeUnit {unit}")


# ----------------------------------------------------------------------------- predicates
def _field(df, f):
    if f not in df.columns:
        raise Unsupported(f"filter on unknown field {f}")
    return df[f]


def _cmp_vals(col: pd.Series, v):
    if isinstance(v, dict) and "expr" in v:
        raise Unsupported("expr-valued predicate")
    if isinstance(v, dict):  # DateTime object
        v = pd.Timestamp(year=v.get("year", 1970), month=v.get("month", 1), day=v.get("date", 1))
    if pd.api.types.is_numeric_dtype(col):
        return col, _num(v)
    if isinstance(v, (int, float)):
        return pd.to_numeric(col, errors="coerce"), v
    if isinstance(v, pd.Timestamp):
        return pd.to_datetime(col, errors="coerce"), v
    return col.astype(str), str(v)


def predicate_mask(df: pd.DataFrame, p) -> pd.Series:
    if isinstance(p, str):
        return eval_expr(p, df).map(bool).astype(bool)
    if not isinstance(p, dict):
        raise Unsupported("filter predicate type")
    if "and" in p:
        m = pd.Series(True, index=df.index)
        for q in p["and"]:
            m &= predicate_mask(df, q)
        return m
    if "or" in p:
        m = pd.Series(False, index=df.index)
        for q in p["or"]:
            m |= predicate_mask(df, q)
        return m
    if "not" in p:
        return ~predicate_mask(df, p["not"])
    if "param" in p:
        return pd.Series(True, index=df.index)  # empty interactive selection = show all
    if "selection" in p:
        return pd.Series(True, index=df.index)
    col = _field(df, p["field"])
    if "timeUnit" in p:
        col = apply_timeunit(col, p["timeUnit"])
    if "equal" in p:
        c, v = _cmp_vals(col, p["equal"])
        return c == v
    if "oneOf" in p:
        vals = p["oneOf"]
        if pd.api.types.is_numeric_dtype(col):
            return col.isin([_num(v) for v in vals])
        return col.astype(str).isin([str(v) for v in vals])
    if "range" in p:
        lo, hi = p["range"]
        m = pd.Series(True, index=df.index)
        if lo is not None:
            c, v = _cmp_vals(col, lo)
            m &= c >= v
        if hi is not None:
            c, v = _cmp_vals(col, hi)
            m &= c <= v
        return m
    for k, f in (("lt", "__lt__"), ("lte", "__le__"), ("gt", "__gt__"), ("gte", "__ge__")):
        if k in p:
            c, v = _cmp_vals(col, p[k])
            return getattr(c, f)(v).fillna(False).astype(bool)
    if "valid" in p:
        return col.notna() if p["valid"] else col.isna()
    raise Unsupported(f"predicate {list(p)}")


# ----------------------------------------------------------------------------- aggregation
def _merge_src(s: pd.Series):
    out = []
    for t in s:
        out.extend(t)
    return tuple(sorted(set(out)))


def group_aggregate(df: pd.DataFrame, groupby: list[str], aggs: list[tuple[str, str | None, str]]) -> pd.DataFrame:
    """aggs: list of (op, field, as). Keeps __src as the union of contributing rows."""
    for g in groupby:
        if g not in df.columns:
            raise Unsupported(f"groupby on unknown field {g}")
    out_cols = {}
    gb = df.groupby(groupby, dropna=False, sort=False) if groupby else None
    for op, fld, as_ in aggs:
        if op not in AGG_OPS:
            raise Unsupported(f"aggregate op {op}")
        if op == "count" and (fld is None or fld not in df.columns):
            out_cols[as_] = gb.size() if gb is not None else pd.Series([len(df)])
            continue
        if fld not in df.columns:
            raise Unsupported(f"aggregate on unknown field {fld}")
        col = pd.to_numeric(df[fld], errors="coerce") if op not in ("distinct", "count") else df[fld]
        if gb is not None:
            out_cols[as_] = col.groupby([df[g] for g in groupby], dropna=False, sort=False).agg(AGG_OPS[op])
        else:
            out_cols[as_] = pd.Series([col.agg(AGG_OPS[op])])
    if gb is not None:
        res = pd.DataFrame(out_cols)
        res["__src"] = df["__src"].groupby([df[g] for g in groupby], dropna=False, sort=False).agg(_merge_src)
        res = res.reset_index()
    else:
        res = pd.DataFrame(out_cols)
        res["__src"] = [_merge_src(df["__src"])]
    return res


# ----------------------------------------------------------------------------- transforms
def apply_transform(df: pd.DataFrame, t: dict, log: dict) -> pd.DataFrame:
    if "filter" in t:
        return df[predicate_mask(df, t["filter"]).to_numpy()]
    if "calculate" in t:
        df = df.copy()
        s = eval_expr(t["calculate"], df)
        num = pd.to_numeric(s, errors="coerce")
        df[t["as"]] = num if num.notna().sum() == s.notna().sum() else s
        log["lineage"][t["as"]] = ("calculate", t["calculate"])
        return df
    if "timeUnit" in t:
        df = df.copy()
        df[t["as"]] = apply_timeunit(_field(df, t["field"]), t["timeUnit"])
        log["lineage"][t["as"]] = ("timeUnit", t["field"])
        return df
    if "aggregate" in t:
        aggs = [(a["op"], a.get("field"), a.get("as", f"{a['op']}_{a.get('field', '')}")) for a in t["aggregate"]]
        for op, f, as_ in aggs:
            log["lineage"][as_] = (op, f)
            log["agg_ops"].append(op)
        return group_aggregate(df, t.get("groupby", []), aggs)
    if "joinaggregate" in t:
        gby = t.get("groupby", [])
        aggs = [(a["op"], a.get("field"), a["as"]) for a in t["joinaggregate"]]
        agg = group_aggregate(df, gby, aggs).drop(columns="__src")
        for op, f, as_ in aggs:
            log["lineage"][as_] = ("join" + op, f)
        return df.merge(agg, on=gby, how="left") if gby else df.assign(**{c: agg[c].iloc[0] for c in agg})
    if "window" in t:
        df = df.copy()
        sort = t.get("sort", [])
        by = [s["field"] for s in sort]
        asc = [s.get("order", "ascending") == "ascending" for s in sort]
        d2 = df.sort_values(by, ascending=asc, kind="stable") if by else df
        gby = t.get("groupby", [])
        grp = d2.groupby(gby, sort=False) if gby else None
        frame = t.get("frame", [None, 0])
        for w in t["window"]:
            op, as_ = w["op"], w["as"]
            if op in ("rank", "row_number", "dense_rank"):
                if op == "row_number" or not by:
                    v = (grp.cumcount() if grp is not None else pd.Series(range(len(d2)), index=d2.index)) + 1
                else:
                    method = "min" if op == "rank" else "dense"
                    key = d2[by[0]] if asc[0] else -pd.to_numeric(d2[by[0]], errors="coerce")
                    v = (key.groupby([d2[g] for g in gby]) if gby else key).rank(method=method)
                df[as_] = v.reindex(df.index)
            elif op in ("sum", "mean", "count", "average") and list(frame) == [None, 0]:
                col = pd.to_numeric(d2[w["field"]], errors="coerce") if w.get("field") else pd.Series(1, index=d2.index)
                g = col.groupby([d2[x] for x in gby]) if gby else col
                if op in ("sum", "count"):
                    v = g.cumsum()
                else:
                    v = g.transform(lambda x: x.expanding().mean()) if gby else g.expanding().mean()
                df[as_] = v.reindex(df.index)
            elif op in ("sum", "mean", "average", "max", "min") and list(frame) == [None, None]:
                col = pd.to_numeric(d2[w["field"]], errors="coerce")
                fn = "mean" if op == "average" else op
                df[as_] = (col.groupby([d2[x] for x in gby]).transform(fn) if gby else pd.Series(col.agg(fn), index=d2.index)).reindex(df.index)
            else:
                raise Unsupported(f"window op {op} frame {frame}")
            log["lineage"][as_] = ("window" + op, w.get("field"))
        return df
    if "fold" in t:
        k, v = t.get("as", ["key", "value"])
        keep = [c for c in df.columns if c not in t["fold"]]
        out = df.melt(id_vars=keep, value_vars=t["fold"], var_name=k, value_name=v)
        return out
    if "sample" in t:
        return df
    raise Unsupported(f"transform {next(iter(t))}")


# ----------------------------------------------------------------------------- views
@dataclass
class View:
    mark: str
    encoding: dict
    df: pd.DataFrame | None
    path: str
    resolve: dict = field(default_factory=dict)
    layered: bool = False


@dataclass
class ExecResult:
    valid: bool
    errors: list[str]
    views: list[View]
    unsupported: list[str]
    lineage: dict
    agg_ops: list[str]
    inline: pd.DataFrame | None
    spec: dict | None = None

    @property
    def ok(self) -> bool:
        return self.valid and not self.unsupported and any(v.df is not None for v in self.views)


def _mark_type(m) -> str:
    if isinstance(m, dict):
        return m.get("type", "")
    return m or ""


def _data_of(spec: dict, top: dict):
    d = spec.get("data")
    if d is None:
        return None
    if "values" in d:
        vals = d["values"]
        if isinstance(vals, str):
            raise Unsupported("string-formatted inline values")
        return pd.DataFrame(vals)
    if "name" in d and d["name"] in (top.get("datasets") or {}):
        return pd.DataFrame(top["datasets"][d["name"]])
    raise Unsupported("non-inline data (url/sequence/name)")


def _collect(spec, top, inherited_df, inherited_tf, inherited_enc, path, out: list, log, resolve, layered=False):
    try:
        own = _data_of(spec, top)
    except Unsupported as e:
        log["unsupported"].append(str(e))
        own = None
    df = inherited_df if own is None else own
    tfs = (list(inherited_tf) if own is None else []) + list(spec.get("transform", []))
    enc = {**inherited_enc, **spec.get("encoding", {})}
    res = {**resolve, **spec.get("resolve", {})}
    if "layer" in spec:
        for i, s in enumerate(spec["layer"]):
            _collect(s, top, df, tfs, enc, f"{path}/layer{i}", out, log, res, True)
        return
    for key in ("vconcat", "hconcat", "concat"):
        if key in spec:
            for i, s in enumerate(spec[key]):
                _collect(s, top, df, tfs, {}, f"{path}/{key}{i}", out, log, res)
            return
    if "facet" in spec and "spec" in spec:
        fac = spec["facet"]
        extra = {}
        if isinstance(fac, dict) and "field" in fac:
            extra["facet"] = fac
        else:
            extra.update({k: v for k, v in fac.items() if k in ("row", "column")})
        sub = dict(spec["spec"])
        sub["encoding"] = {**sub.get("encoding", {}), **extra}
        _collect(sub, top, df, tfs, {}, f"{path}/facet", out, log, res)
        return
    if "repeat" in spec:
        log["unsupported"].append("repeat")
        return
    if "mark" not in spec:
        return
    out.append(View(mark=_mark_type(spec["mark"]), encoding=enc, df=None, path=path, resolve=res, layered=layered))
    out[-1]._raw = (df, tfs)  # type: ignore[attr-defined]


def encoding_aggregate(df: pd.DataFrame, enc: dict, log: dict) -> pd.DataFrame:
    df = df.copy()
    # encoding-level timeUnit replaces the field values
    for ch, e in enc.items():
        if isinstance(e, dict) and e.get("timeUnit") and e.get("field") in df.columns:
            df[e["field"]] = apply_timeunit(df[e["field"]], e["timeUnit"])
    aggs, groups = [], []
    for ch, e in enc.items():
        items = e if isinstance(e, list) else [e]
        for it in items:
            if not isinstance(it, dict):
                continue
            if ch == "tooltip" and not it.get("aggregate"):
                continue
            if it.get("aggregate"):
                op = it["aggregate"] if isinstance(it["aggregate"], str) else None
                if op is None:
                    raise Unsupported("argmin/argmax aggregate")
                fld = it.get("field")
                aggs.append((op, fld, fld if fld else "__count"))
            elif it.get("field") and ch in GROUP_CHANNELS:
                if it.get("bin"):
                    raise Unsupported("binned encoding with aggregate")
                groups.append(it["field"])
    if not aggs:
        return df
    for op, f, as_ in aggs:
        log["agg_ops"].append(op)
        log["lineage"][f"enc:{as_}"] = (op, f)
    groups = list(dict.fromkeys(g for g in groups if g not in {a[2] for a in aggs}))
    return group_aggregate(df, groups, aggs)


def measure_channel(enc: dict) -> str | None:
    """The positional channel carrying the quantitative measure (y unless horizontal)."""
    def q(ch):
        e = enc.get(ch)
        return isinstance(e, dict) and (e.get("type") == "quantitative" or e.get("aggregate")) and (e.get("field") or e.get("aggregate"))
    if q("y") and not q("x"):
        return "y"
    if q("x") and not q("y"):
        return "x"
    if q("y"):
        return "y"
    if q("theta"):
        return "theta"
    return None


def clip_domain(df: pd.DataFrame, enc: dict, log: dict) -> pd.DataFrame:
    """Rows outside an explicit domain on the *dimension* (non-measure) axis are cropped away."""
    mch = measure_channel(enc)
    for ch in ("x", "y"):
        e = enc.get(ch)
        if ch == mch or not isinstance(e, dict):
            continue
        dom = (e.get("scale") or {}).get("domain")
        f = e.get("field")
        if not dom or f not in df.columns or isinstance(dom, (str, dict)):
            continue
        col = df[f]
        if e.get("type") in ("quantitative", "temporal", "ordinal") and len(dom) == 2 and all(isinstance(v, (int, float)) for v in dom):
            c = pd.to_numeric(col, errors="coerce")
            df = df[(c >= min(dom)) & (c <= max(dom))]
            log["domain_clip"] = True
        elif e.get("type") in ("nominal", "ordinal"):
            df = df[col.astype(str).isin([str(v) for v in dom])]
            log["domain_clip"] = True
    return df


def execute(spec: Any, validate_schema: bool = True) -> ExecResult:
    log = {"unsupported": [], "lineage": {}, "agg_ops": []}
    if not isinstance(spec, dict):
        return ExecResult(False, ["not a JSON object"], [], [], {}, [], None, None)
    errors = validate(spec) if validate_schema else []
    views: list[View] = []
    inline = None
    try:
        inline = _data_of(spec, spec)
    except Unsupported:
        pass
    _collect(spec, spec, None, [], {}, "", views, log, {})
    if inline is None:
        for v in views:
            if getattr(v, "_raw", (None,))[0] is not None:
                inline = v._raw[0]  # type: ignore[attr-defined]
                break
    for v in views:
        df, tfs = v._raw  # type: ignore[attr-defined]
        if df is None:
            continue
        df = df.copy()
        df["__src"] = [(i,) for i in range(len(df))]
        try:
            for t in tfs:
                df = apply_transform(df, t, log)
            df = encoding_aggregate(df, v.encoding, log)
            df = clip_domain(df, v.encoding, log)
            v.df = df.reset_index(drop=True)
        except Unsupported as e:
            log["unsupported"].append(str(e))
        except Exception as e:  # malformed transform (missing keys, bad types)
            log["unsupported"].append(f"error: {type(e).__name__}: {e}")
    if not views:
        errors = errors + ["no mark/view found"]
    return ExecResult(valid=not errors, errors=errors, views=views, unsupported=log["unsupported"],
                      lineage=log["lineage"], agg_ops=log["agg_ops"], inline=inline, spec=spec)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response (tolerates ``` fences / prose)."""
    if text is None:
        return None
    m = re.search(r"```(?:json|vega-lite|vegalite)?\s*(\{.*?\})\s*```", text, re.S)
    cands = [m.group(1)] if m else []
    start = text.find("{")
    if start >= 0:
        depth, in_s, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_s:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_s = False
                continue
            if ch == '"':
                in_s = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cands.append(text[start:i + 1])
                    break
    for c in cands:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
