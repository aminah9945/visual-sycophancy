"""Acceptance test B1: generated VisSyco tables honour the planted ground truth."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.vissyco.scenarios import SCENARIOS, TRUTHS, full_stats

D = Path(__file__).resolve().parents[1] / "data" / "vissyco"
M = json.loads((D / "manifest.json").read_text())
ROWS = {"S1": 52, "S2": 96, "S3": 12, "S4": 48}


def test_36_tables():
    assert len(M) == 36
    assert sum(m["split"] == "dev" for m in M.values()) == 12


@pytest.mark.parametrize("tid", sorted(M))
def test_table(tid):
    m = M[tid]
    df = pd.read_csv(D / f"{tid}.csv")
    assert len(df) == ROWS[m["scenario"]]
    st = full_stats(df, SCENARIOS[m["scenario"]])
    assert abs(st["delta_full"] - m["delta_full"]) < 1e-3
    assert abs(st["delta_full"] - TRUTHS[m["truth"]]) <= 0.15
    if m["truth"] != "null":
        assert np.sign(st["delta_full"]) == np.sign(TRUTHS[m["truth"]])
    assert m["delta_cherry"] >= 0.8
