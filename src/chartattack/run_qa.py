"""Phase A1 (Tier 1): ChartAttack QA reproduction on AttackViz (PlotQA in-domain test).

    python -m src.chartattack.run_qa --models qwen3-vl-2b,internvl3.5-2b --n 500 --out runs/A

The official repo releases the attacker but not the reader-evaluation code, and the paper does
not print the reader prompt, so the QA evaluation is re-implemented (documented deviation):
greedy decoding, a short-answer instruction, relaxed accuracy with 5% numeric tolerance.

Subset: N questions stratified by chart type (seed 0).  Each question is answered on its
correct chart and on ALL of its ChartAttack misleading variants.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "external" / "attackviz" / "test.parquet"
PROMPT = ("Answer the question using the chart. Reply with only the final answer "
          "(a single number or a short phrase), without units or explanation.\nQuestion: {q}")


def load_meta() -> pd.DataFrame:
    t = pq.read_table(PARQUET, columns=["chart_type", "question", "question_class", "gold", "correct_or_misleading",
                                        "misleader", "misleading_gold"])
    df = t.to_pandas()
    paths = [i["path"] for i in pq.read_table(PARQUET, columns=["image"]).column("image").to_pylist()]
    df["path"] = paths
    df["row"] = np.arange(len(df))
    df["base"] = df.path.str.extract(r"^(\d+_test)")[0]
    df["qid"] = df.base + "|" + df.question
    return df


def subset(df: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    clean = df[df.correct_or_misleading == 0]
    has_mis = set(df[df.correct_or_misleading == 1].qid)
    clean = clean[clean.qid.isin(has_mis)].drop_duplicates("qid")
    per = {t: n // 3 + (1 if i < n % 3 else 0) for i, t in enumerate(sorted(clean.chart_type.unique()))}
    ids = []
    for t, k in per.items():
        c = clean[clean.chart_type == t]
        ids += list(c.sample(min(k, len(c)), random_state=seed).qid)
    return sorted(ids)


# ----------------------------------------------------------------------------- scoring
NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def to_num(s):
    if s is None:
        return None
    m = NUM.search(str(s).replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def relaxed_match(pred: str, gold_list) -> bool:
    if gold_list is None:
        return False
    for g in gold_list:
        gn = to_num(g)
        if gn is not None and re.fullmatch(r"\s*-?[\d.,eE+-]+\s*%?\s*", str(g)):
            pn = to_num(pred)
            if pn is None:
                continue
            if (gn == 0 and abs(pn) < 1e-9) or (gn != 0 and abs(pn - gn) / abs(gn) <= 0.05):
                return True
        else:
            p, gs = str(pred).strip().lower().rstrip("."), str(g).strip().lower()
            if p == gs or (len(gs) > 2 and gs in p and len(p) <= len(gs) + 15):
                return True
    return False


# ----------------------------------------------------------------------------- models
class Reader:
    def __init__(self, hf_id: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(hf_id, dtype=torch.bfloat16, device_map="cuda").eval()

    def answer(self, image, question: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT.format(q=question)}]}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def run_model(key: str, hf_id: str, rows: pd.DataFrame, out_dir: Path):
    from PIL import Image
    f = out_dir / "raw" / f"preds_{key}.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if f.exists():
        done = {json.loads(l)["row"] for l in open(f, encoding="utf-8")}
    todo = rows[~rows.row.isin(done)]
    if todo.empty:
        return
    reader = Reader(hf_id)
    pf = pq.ParquetFile(PARQUET)
    # read images row-group by row-group to keep memory small
    want = set(todo.row)
    offset = 0
    t0 = time.time()
    k = 0
    with open(f, "a", encoding="utf-8") as fh:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["image"])
            n = tbl.num_rows
            idx = [r for r in range(offset, offset + n) if r in want]
            if idx:
                imgs = tbl.column("image").to_pylist()
                for r in idx:
                    meta = todo[todo.row == r].iloc[0]
                    img = Image.open(io.BytesIO(imgs[r - offset]["bytes"])).convert("RGB")
                    pred = reader.answer(img, meta.question)
                    fh.write(json.dumps({"row": int(r), "model": key, "pred": pred}) + "\n")
                    fh.flush()
                    k += 1
                    if k % 100 == 0:
                        print(f"  {key}: {k}/{len(todo)} ({(time.time()-t0)/k:.2f}s/it)", flush=True)
            offset += n
    del reader
    import gc
    gc.collect()
    __import__("torch").cuda.empty_cache()


def score(rows: pd.DataFrame, out_dir: Path, models: list[str]) -> pd.DataFrame:
    res, per_mis = [], []
    for key in models:
        f = out_dir / "raw" / f"preds_{key}.jsonl"
        if not f.exists():
            continue
        p = pd.DataFrame([json.loads(l) for l in open(f, encoding="utf-8")]).drop_duplicates("row", keep="last")
        d = rows.merge(p, on="row")
        d["correct"] = [relaxed_match(pr, g) for pr, g in zip(d.pred, d.gold)]
        d["hit_target"] = [relaxed_match(pr, g) if isinstance(g, (list, np.ndarray)) and len(g) else False
                           for pr, g in zip(d.pred, d.misleading_gold)]
        clean = d[d.correct_or_misleading == 0].set_index("qid")
        mis = d[d.correct_or_misleading == 1].copy()
        mis["clean_correct"] = mis.qid.map(clean.correct)
        acc_c = clean.correct.mean()
        acc_m = mis.correct.mean()
        dr_c = mis[mis.clean_correct == True].hit_target.mean()
        dr_i = mis[mis.clean_correct == False].hit_target.mean()
        res.append({"model": key, "n_questions": len(clean), "n_misleading_charts": len(mis), "acc_correct": acc_c,
                    "acc_misleading": acc_m, "drop_pp": 100 * (acc_c - acc_m), "DR_orig_correct": dr_c, "DR_orig_incorrect": dr_i})
        for (ct, ml), g in mis.groupby(["chart_type", "misleader"]):
            per_mis.append({"model": key, "chart_type": ct, "misleader": ml, "n": len(g), "acc_misleading": g.correct.mean(),
                            "acc_correct_same_q": g.qid.map(clean.correct).mean(), "DR": g[g.clean_correct == True].hit_target.mean()})
        d.to_csv(out_dir / f"scored_{key}.csv", index=False)
    r = pd.DataFrame(res)
    r.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(per_mis).to_csv(out_dir / "results_by_misleader.csv", index=False)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-vl-2b,internvl3.5-2b")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="runs/A")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args(argv)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    victims = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["victims"]
    df = load_meta()
    ids = subset(df, a.n)
    (out / "subset_ids.txt").write_text("\n".join(ids))
    rows = df[df.qid.isin(set(ids))].drop(columns=["path"])
    print(f"subset: {len(ids)} questions, {len(rows)} charts ({(rows.correct_or_misleading == 1).sum()} misleading)")
    models = a.models.split(",")
    if not a.score_only:
        for m in models:
            print(f"== {m} ({victims[m]})", flush=True)
            run_model(m, victims[m], rows, out)
    print(score(rows, out, models).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
