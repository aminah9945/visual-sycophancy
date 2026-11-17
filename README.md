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

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install requests
python litreview/lit.py run-queries --since 2019
python litreview/lit.py verify paper/main.tex
```

Experiment code (`src/`, `dashboard/`) is added phase by phase, following `docs/EXPERIMENT_PLAN.md`.

## Reproducing results
One command per table will be listed here as experiments are completed.
