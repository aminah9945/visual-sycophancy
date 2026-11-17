"""Acceptance test B3: >= 15 hand-written specs with hand-computed outputs."""
import pytest

from src.executor.vl_exec import execute, extract_json, validate

# 8 rows: week 1..8, sales = 10,20,...,80, region alternating N/S
VALS = [{"week": w, "sales": 10 * w, "region": "N" if w % 2 else "S"} for w in range(1, 9)]


def spec(transform=None, encoding=None, mark="bar", **kw):
    s = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "data": {"values": VALS}, "mark": mark,
         "encoding": encoding or {"x": {"field": "week", "type": "ordinal"}, "y": {"field": "sales", "type": "quantitative"}}}
    if transform:
        s["transform"] = transform
    s.update(kw)
    return s


def run(s):
    ex = execute(s)
    assert ex.valid, ex.errors
    assert not ex.unsupported, ex.unsupported
    return ex.views[0].df


def test_01_identity():
    df = run(spec())
    assert len(df) == 8 and df.sales.mean() == 45


def test_02_filter_gte_predicate():
    df = run(spec([{"filter": {"field": "week", "gte": 5}}]))
    assert len(df) == 4 and df.sales.mean() == 65


def test_03_filter_expression_string():
    df = run(spec([{"filter": "datum.week >= 3 && datum.week <= 5"}]))
    assert list(df.week) == [3, 4, 5] and df.sales.mean() == 40


def test_04_filter_range():
    df = run(spec([{"filter": {"field": "week", "range": [2, 4]}}]))
    assert len(df) == 3 and df.sales.sum() == 90


def test_05_filter_oneof_and_equal():
    assert len(run(spec([{"filter": {"field": "region", "oneOf": ["N"]}}]))) == 4
    df = run(spec([{"filter": {"field": "region", "equal": "S"}}]))
    assert df.sales.mean() == 50  # 20,40,60,80


def test_06_filter_logical_composition():
    df = run(spec([{"filter": {"and": [{"field": "week", "gt": 2}, {"not": {"field": "region", "equal": "N"}}]}}]))
    assert list(df.week) == [4, 6, 8]


def test_07_calculate_then_aggregate():
    df = run(spec([{"calculate": "datum.sales * 2", "as": "s2"}, {"aggregate": [{"op": "mean", "field": "s2", "as": "m"}]}],
                  encoding={"y": {"field": "m", "type": "quantitative"}}))
    assert len(df) == 1 and df.m.iloc[0] == 90


def test_08_aggregate_groupby():
    df = run(spec([{"aggregate": [{"op": "sum", "field": "sales", "as": "tot"}], "groupby": ["region"]}],
                  encoding={"x": {"field": "region", "type": "nominal"}, "y": {"field": "tot", "type": "quantitative"}}))
    d = dict(zip(df.region, df.tot))
    assert d == {"N": 160, "S": 200}


def test_09_encoding_aggregate_mean():
    df = run(spec(encoding={"x": {"field": "region", "type": "nominal"},
                            "y": {"field": "sales", "aggregate": "mean", "type": "quantitative"}}))
    d = dict(zip(df.region, df.sales))
    assert d == {"N": 40, "S": 50}
    assert sorted(len(s) for s in df["__src"]) == [4, 4]


def test_10_encoding_count():
    df = run(spec(encoding={"x": {"field": "region", "type": "nominal"}, "y": {"aggregate": "count", "type": "quantitative"}}))
    assert set(df["__count"]) == {4}


def test_11_window_rank_topk():
    df = run(spec([{"window": [{"op": "rank", "as": "r"}], "sort": [{"field": "sales", "order": "descending"}]},
                   {"filter": "datum.r <= 3"}]))
    assert sorted(df.week) == [6, 7, 8]


def test_12_window_cumulative_sum():
    df = run(spec([{"window": [{"op": "sum", "field": "sales", "as": "cum"}], "sort": [{"field": "week"}], "frame": [None, 0]}]))
    assert df.sort_values("week").cum.iloc[-1] == 360


def test_13_x_domain_clipping_crops_window():
    s = spec(encoding={"x": {"field": "week", "type": "quantitative", "scale": {"domain": [3, 6]}},
                       "y": {"field": "sales", "type": "quantitative"}}, mark="line")
    df = run(s)
    assert list(df.week) == [3, 4, 5, 6]


def test_14_layer_inherits_data_and_transform():
    s = {"data": {"values": VALS}, "transform": [{"filter": {"field": "week", "lte": 2}}],
         "layer": [{"mark": "bar", "encoding": {"x": {"field": "week", "type": "ordinal"}, "y": {"field": "sales", "type": "quantitative"}}},
                   {"mark": "text", "encoding": {"x": {"field": "week", "type": "ordinal"}, "y": {"field": "sales", "type": "quantitative"}, "text": {"field": "sales"}}}]}
    ex = execute(s)
    assert ex.valid and len(ex.views) == 2 and all(len(v.df) == 2 for v in ex.views)


def test_15_vconcat_views():
    s = {"data": {"values": VALS}, "vconcat": [
        {"transform": [{"filter": "datum.week > 6"}], "mark": "line", "encoding": {"x": {"field": "week", "type": "quantitative"}, "y": {"field": "sales", "type": "quantitative"}}},
        {"mark": "line", "encoding": {"x": {"field": "week", "type": "quantitative"}, "y": {"field": "sales", "type": "quantitative"}}}]}
    ex = execute(s)
    assert [len(v.df) for v in ex.views] == [2, 8]


def test_16_invalid_schema_and_unsupported():
    assert validate({"mark": "bar", "encoding": {"x": {"field": "a", "type": "banana"}}})
    ex = execute(spec([{"loess": "sales", "on": "week"}]))
    assert ex.unsupported


def test_17_url_data_unsupported():
    ex = execute({"data": {"url": "data.csv"}, "mark": "bar", "encoding": {"x": {"field": "a", "type": "nominal"}}})
    assert ex.unsupported and not ex.ok


def test_18_expression_is_sandboxed():
    ex = execute(spec([{"filter": "__import__('os').system('echo hi')"}]))
    assert ex.unsupported


def test_19_extract_json_from_fenced_text():
    t = 'Here you go:\n```json\n{"mark": "bar", "data": {"values": [{"a": 1}]}}\n```\nEnjoy'
    assert extract_json(t)["mark"] == "bar"
    assert extract_json('prefix {"a": {"b": "}"}} suffix') == {"a": {"b": "}"}}


def test_20_facet():
    s = {"data": {"values": VALS}, "facet": {"field": "region", "type": "nominal"},
         "spec": {"mark": "bar", "encoding": {"x": {"field": "week", "type": "ordinal"}, "y": {"field": "sales", "aggregate": "sum", "type": "quantitative"}}}}
    ex = execute(s)
    assert ex.valid, ex.errors
    df = ex.views[0].df
    assert len(df) == 8 and df.sales.sum() == 360
