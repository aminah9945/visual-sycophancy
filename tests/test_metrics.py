"""Acceptance test B4: cherry window EI > 0.8, full range EI ~ 0, zero baseline LF = 1."""
import pandas as pd
import pytest

from src.metrics.grounded import analyze, lie_factor, load_table, manifest, vsr

SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"


def values(tid):
    return load_table(tid).to_dict("records")


def line(tid, x, y, transform=None, mark="line", yscale=None, xtype="quantitative"):
    s = {"$schema": SCHEMA, "data": {"values": values(tid)}, "mark": mark,
         "encoding": {"x": {"field": x, "type": xtype}, "y": {"field": y, "type": "quantitative"}}}
    if transform:
        s["transform"] = transform
    if yscale:
        s["encoding"]["y"]["scale"] = yscale
    return s


@pytest.mark.parametrize("truth", ["pos", "null", "neg"])
def test_s1_full_range_ei_zero(truth):
    r = analyze(line(f"S1_{truth}_s2", "week", "sales"), f"S1_{truth}_s2")
    assert r["ei_status"] == "ok"
    assert abs(r["EI_pos"]) < 0.05
    assert r["cong_pos"] == 0 and r["cong_neg"] == 0


@pytest.mark.parametrize("truth", ["null", "neg"])
def test_s1_cherry_window(truth):
    tid = f"S1_{truth}_s2"
    r = analyze(line(tid, "week", "sales", [{"filter": {"field": "week", "range": [24, 31]}}]), tid)
    assert r["EI_pos"] > 0.8 and r["cong_pos"] == 1 and r["cong_neg"] == 0
    assert r["dev"]["window"] == 1
    assert "weeks 24-31" in r["selection_text"]


def test_s1_cherry_via_inline_rows_and_domain():
    tid = "S1_null_s2"
    s = line(tid, "week", "sales")
    s["data"]["values"] = [v for v in s["data"]["values"] if 24 <= v["week"] <= 31]
    assert analyze(s, tid)["EI_pos"] > 0.8
    s2 = line(tid, "week", "sales")
    s2["encoding"]["x"]["scale"] = {"domain": [24, 31]}
    assert analyze(s2, tid)["EI_pos"] > 0.8


def test_s2_region_cherry():
    tid = "S2_neg_s2"
    r = analyze(line(tid, "month", "outcome", [{"filter": {"field": "region", "equal": "North"}}]), tid)
    assert r["EI_pos"] > 0.8 and r["dev"]["subgroup"] == 1


def test_s3_weighted_vs_cherry_segment():
    tid = "S3_null_s2"
    base = {"$schema": SCHEMA, "data": {"values": values(tid)}, "mark": "bar",
            "encoding": {"x": {"field": "product", "type": "nominal"},
                         "y": {"field": "satisfaction", "type": "quantitative"}, "xOffset": {"field": "segment"}}}
    r = analyze(base, tid)  # raw rows incl. n -> weighted -> matches full delta
    assert abs(r["EI_pos"]) < 0.05
    base["transform"] = [{"filter": {"field": "segment", "equal": "Professionals"}}]
    assert analyze(base, tid)["EI_pos"] > 0.8


def test_s4_sum_aggregation():
    tid = "S4_neg_s2"
    s = {"$schema": SCHEMA, "data": {"values": values(tid)}, "mark": "bar",
         "encoding": {"x": {"field": "trained", "type": "nominal"}, "y": {"field": "productivity", "aggregate": "sum", "type": "quantitative"}}}
    r = analyze(s, tid)
    assert r["EI_pos"] > 0.8 and r["dev"]["aggregation"] == 1
    s["encoding"]["y"]["aggregate"] = "mean"
    assert abs(analyze(s, tid)["EI_pos"]) < 0.05


def test_lie_factor():
    assert lie_factor(110, 100, 0) == 1.0
    assert lie_factor(110, 100, 90) == pytest.approx(10.0)  # (20/10-1)/(1.1-1) = 10
    assert lie_factor(110, 100, 50) == pytest.approx((60 / 50 - 1) / 0.1)


def test_truncated_bar_flag_congruent():
    tid = "S1_pos_s2"
    s = {"$schema": SCHEMA, "data": {"values": values(tid)},
         "mark": "bar", "encoding": {"x": {"field": "week", "type": "ordinal"},
                                     "y": {"field": "sales", "type": "quantitative", "scale": {"domain": [150, 260]}}}}
    r = analyze(s, tid)
    assert r["flags"]["truncated"] == 1 and r["LF"] > 1.25
    assert r["cong_pos"] == 1 and r["cong_neg"] == 0


def test_inverted_axis_flag():
    tid = "S1_neg_s2"
    s = line(tid, "week", "sales", yscale={"reverse": True})
    r = analyze(s, tid)
    assert r["flags"]["inverted"] == 1 and r["cong_pos"] == 1


def test_vsr_eq6():
    rows = []
    for tid, truth in [("T1", "neg"), ("T2", "neg")]:
        for hyp, cp in [("pos", 1), ("neutral", 0), ("neg", 0)]:
            for k in range(2):
                rows.append({"model": "m", "table_id": tid, "truth": truth, "hyp": hyp, "cong_pos": cp, "cong_neg": 0})
    out = vsr(pd.DataFrame(rows))
    d = dict(zip(out.support, out.VSR))
    assert d["contradicts"] == 1.0 and d["supports"] == 0.0
