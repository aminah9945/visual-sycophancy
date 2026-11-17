"""Phase C1: one LLM entry point with a disk cache, retries, cost tracking and a hard budget.

    generate(model_key, messages, temperature, sample_idx) -> dict

Cache key = (model id, prompt hash, temperature, sample_idx, max_tokens): re-running never pays twice.
Spend is accumulated in runs/cache/spend.json and calls stop once configs/models.yaml:budget_usd is hit.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "runs" / "cache"
CFG = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
_lock = threading.Lock()
_db = None
_client = None


class BudgetExceeded(RuntimeError):
    pass


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.split("#")[0].strip().strip('"').strip("'"))


def _conn():
    global _db
    if _db is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(CACHE_DIR / "llm_cache.sqlite", check_same_thread=False)
        _db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        _db.commit()
    return _db


def _spend_file():
    return CACHE_DIR / "spend.json"


def spend() -> float:
    f = _spend_file()
    return json.loads(f.read_text())["usd"] if f.exists() else 0.0


def _add_spend(usd: float):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s = spend() + usd
    _spend_file().write_text(json.dumps({"usd": s}))
    return s


def _openrouter():
    global _client
    if _client is None:
        _load_env()
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY missing: put it in .env (see EXPERIMENT_PLAN.md 1.3)")
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=0, timeout=180)
    return _client


def model_cfg(model_key: str) -> dict:
    g = CFG["generators"]
    if model_key in g:
        return {"key": model_key, **g[model_key]}
    for k, v in g.items():
        if v["id"] == model_key:
            return {"key": k, **v}
    raise KeyError(model_key)


def cache_key(model_id, messages, temperature, sample_idx, max_tokens) -> str:
    blob = json.dumps([model_id, messages, temperature, sample_idx, max_tokens], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def generate(model_key: str, messages: list[dict], temperature: float | None = None, sample_idx: int = 0,
             max_tokens: int | None = None, seed: int | None = None) -> dict:
    mc = model_cfg(model_key)
    dec = CFG["decoding"]
    temperature = dec["temperature"] if temperature is None else temperature
    max_tokens = max_tokens or dec["max_tokens"]
    k = cache_key(mc["id"], messages, temperature, sample_idx, max_tokens)
    with _lock:
        row = _conn().execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
    if row:
        out = json.loads(row[0])
        out["cached"] = True
        return out
    with _lock:
        if spend() >= CFG["budget_usd"]:
            raise BudgetExceeded(f"budget {CFG['budget_usd']} USD reached")
    t0 = time.time()
    last = None
    for attempt in range(6):
        try:
            r = _openrouter().chat.completions.create(
                model=mc["id"], messages=messages, temperature=temperature, top_p=dec["top_p"],
                max_tokens=max_tokens, seed=(seed if seed is not None else 1000 + sample_idx),
                extra_body={"usage": {"include": True}},
            )
            if not r.choices:
                raise RuntimeError(f"empty choices: {getattr(r, 'error', None)}")
            text = r.choices[0].message.content or ""
            u = r.usage
            tin, tout = (u.prompt_tokens, u.completion_tokens) if u else (None, None)
            cost = getattr(u, "cost", None) if u else None
            if cost is None and tin is not None:
                cost = (tin * mc["price_in"] + tout * mc["price_out"]) / 1e6
            out = {"response": text, "tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
                   "latency_s": round(time.time() - t0, 2), "finish_reason": r.choices[0].finish_reason,
                   "model_id": mc["id"], "provider": getattr(r, "provider", None), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with _lock:
                _add_spend(cost or 0.0)
                _conn().execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(out)))
                _conn().commit()
            out["cached"] = False
            return out
        except Exception as e:  # rate limits, provider errors, timeouts
            last = e
            time.sleep(min(60, 2 ** (attempt + 1)))
    raise RuntimeError(f"{mc['id']} failed after retries: {last}")
