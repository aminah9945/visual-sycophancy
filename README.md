# Visual Sycophancy: Measuring and Mitigating Hypothesis-Congruent Chart Generation in LLMs

Course research project (Visual Analytics + Generative AI).

**Question:** when a benign user states a belief ("I think the campaign boosted sales"), do LLM chart generators bend their data selection and visual encoding toward it? And can a data-grounded audit (CGCA) prevent it?

| Path | Contents |
|---|---|
| `paper/` | IEEE-format manuscript (`main.tex`, `main.pdf`) |
| `docs/EXPERIMENT_PLAN.md` | Full plan: setup, datasets, experiments, commands, timeline, presentation kit |
| `docs/LIT_REVIEW.md` | Systematic literature review protocol and commands |
| `litreview/lit.py` | Search, snowballing, screening stats, reference verification, BibTeX export |
| `litreview/queries.txt` | Search strings |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # Phase A only
pip install transformers accelerate                                    # Phase A only
echo "OPENROUTER_API_KEY=sk-or-..." > .env                             # never commit
pytest -q                                                              # 77 acceptance tests
```

## Code map
| Path | What |
|---|---|
| `src/vissyco/` | scenarios (Eq. 2), table generator, prompt templates |
| `src/executor/vl_exec.py` | Vega-Lite v5 validation + transform executor (sandboxed expressions) |
| `src/metrics/grounded.py` | Delta, EI, LF, flags, congruence, VSR (Eqs. 2-6) |
| `src/metrics/stats.py` | cluster bootstrap, clustered logit, McNemar + Holm |
| `src/llm/client.py` | OpenRouter client, disk cache, budget stop |
| `src/cgca/` | CGCA (Algorithm 1) and baselines B1-B3 |
| `src/chartattack/run_qa.py` | ChartAttack Tier-1 QA reproduction on AttackViz |
| `dashboard/app.py` | VA auditing interface (V1-V4) |

## Reproducing results (one command per table)
| Paper item | Command |
|---|---|
| VisSyco data, Table (statistics), Fig. (scenarios) | `python -m src.vissyco.generate_data --out data/vissyco --seeds 1 2 3` then `python -m src.analysis.make_tables --exp DATA` |
| ChartAttack reproduction table | `python -m src.chartattack.run_qa --models qwen3-vl-2b,internvl3.5-2b --n 500 --out runs/A` then `python -m src.analysis.make_tables --exp A` |
| E1 VSR table + figure | `python -m src.run_generation --exp E1 --split test --models all --samples 5` then `python -m src.analysis.make_tables --exp E1` |
| Threshold calibration (dev only) | `python -m src.run_generation --exp DEV --split dev` then `python -m src.calibrate --split dev --eps 0.05` |
| E2 defence comparison + McNemar | `python -m src.run_defenses --exp E2 --split test --defenses B0,B1,B2,B3,CGCA` then `python -m src.analysis.make_tables --exp E2` |
| Ablations A1-A6 + tau curve | `python -m src.run_defenses --exp ABL --split test --ablations A1,A2,A3,A4,A5,A6 --models qwen3-30b,gpt-4o-mini` then `python -m src.analysis.make_tables --exp ABL` |
| Dashboard / case studies | `streamlit run dashboard/app.py` |

All LLM calls are cached in `runs/cache/` (keyed by model, messages, temperature, sample), so reruns are free.
Deviations from the plan are documented in `runs/ENVIRONMENT.md` and `runs/A/DEVIATIONS.md`.

## Literature review
```bash
python litreview/lit.py run-queries --since 2019
python litreview/lit.py verify paper/main.tex
```
