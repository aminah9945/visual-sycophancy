"""CGCA / baselines end-to-end with a mock generator (no API calls)."""
import json

import pytest

import src.cgca.audit as audit
import src.cgca.baselines as baselines
from src.executor.vl_exec import execute
from src.metrics.grounded import analyze, load_table
from src.pipeline import fidelity, make_record
from src.vissyco.prompts import messages_for

TID = "S1_neg_s2"
VALS = load_table(TID).to_dict("records")
SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"


def chart(transform=None):
    s = {"$schema": SCHEMA, "data": {"values": VALS}, "mark": "line",
         "encoding": {"x": {"field": "week", "type": "quantitative"}, "y": {"field": "sales", "type": "quantitative"}}}
    if transform:
        s["transform"] = transform
    return s


CHERRY = chart([{"filter": {"field": "week", "range": [24, 31]}}])
HONEST = chart()


def fake_gen(sequence):
    calls = []

    def gen(model, messages, sample_idx=0, temperature=None, **kw):
        calls.append(messages)
        spec = sequence[min(len(calls) - 1, len(sequence) - 1)]
        return {"response": "```json\n" + json.dumps(spec) + "\n```", "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.0}
    gen.calls = calls
    return gen


def rec0(spec, hyp="pos"):
    msgs = messages_for(load_table(TID), "S1", hyp)
    g = {"response": json.dumps(spec)}
    return make_record("T", TID, hyp, "mock", 0, "none", msgs, g), msgs


def test_cgca_revises_cherry_pick(monkeypatch):
    gen = fake_gen([HONEST])
    monkeypatch.setattr(audit, "generate", gen)
    r, msgs = rec0(CHERRY)
    assert r["congruent"] == 1
    out = audit.run_cgca(r, msgs, "mock", 0.3, 1.25, R=2)
    assert out["revised"] and out["rounds_used"] == 1 and not out["augmented"]
    fb = out["feedbacks"][0]
    assert "weeks 24-31" in fb and "lower for after the campaign" in fb
    assert out["trajectory"][0] > 0.8 and abs(out["trajectory"][1]) < 0.05


def test_cgca_accepts_honest(monkeypatch):
    gen = fake_gen([HONEST])
    monkeypatch.setattr(audit, "generate", gen)
    r, msgs = rec0(HONEST)
    out = audit.run_cgca(r, msgs, "mock", 0.3, 1.25)
    assert not out["revised"] and not gen.calls


def test_cgca_augments_when_model_persists(monkeypatch):
    monkeypatch.setattr(audit, "generate", fake_gen([CHERRY]))
    r, msgs = rec0(CHERRY)
    out = audit.run_cgca(r, msgs, "mock", 0.3, 1.25, R=2)
    assert out["augmented"] and out["rounds_used"] == 2
    ex = execute(out["spec"])
    assert ex.valid, ex.errors
    a = analyze(out["spec"], TID)
    assert abs(a["EI_pos"]) < 0.05  # the overview restores full context
    assert fidelity(ex, "S1") == 1


def test_cgca_schema_repair(monkeypatch):
    monkeypatch.setattr(audit, "generate", fake_gen([HONEST]))
    r, msgs = rec0({"mark": "bar", "encoding": {"x": {"field": "week", "type": "banana"}}})
    assert not r["valid"]
    out = audit.run_cgca(r, msgs, "mock", 0.3, 1.25)
    assert out["violations"][0] == ["INVALID"] and analyze(out["spec"], TID)["valid"]


def test_neutral_uses_agnostic_audit(monkeypatch):
    monkeypatch.setattr(audit, "generate", fake_gen([HONEST]))
    r, msgs = rec0(CHERRY, hyp="neutral")
    out = audit.run_cgca(r, msgs, "mock", 0.3, 1.25)
    assert out["flagged_r0"]


def test_linter_fixes_encodings_not_data():
    s = chart([{"filter": {"field": "week", "range": [24, 31]}}])
    s["mark"] = "bar"
    s["encoding"]["y"]["scale"] = {"zero": False, "reverse": True}
    fixed, log = baselines.lint_fix(s)
    assert set(log) == {"truncated", "inverted"}
    a = analyze(fixed, TID)
    assert a["flags"]["truncated"] == 0 and a["flags"]["inverted"] == 0
    assert a["EI_pos"] > 0.8  # linter is blind to the cherry-picked window


def test_self_refine_stops_on_no_issues(monkeypatch):
    def gen(model, messages, **kw):
        return {"response": "NO ISSUES", "tokens_in": 1, "tokens_out": 1, "cost_usd": 0}
    monkeypatch.setattr(baselines, "generate", gen)
    r, msgs = rec0(CHERRY)
    out = baselines.self_refine(r, msgs, "mock")
    assert not out["revised"] and out["extra_calls"] == 1
