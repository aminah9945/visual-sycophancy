# Phase A: deviations from the original ChartAttack setup

Original code: https://github.com/insait-institute/chartAttack (cloned 2026-11-17, `external/chartattack`).
Original data: AttackViz test split, https://huggingface.co/datasets/INSAIT-Institute/AttackViz (`data/test-00000-of-00001.parquet`).
Original requirements: Python 3.12.8, CUDA 11.8 (`requirements.txt` in the repo).

1. **No reader-evaluation code is released.** The repository contains only the demonstration-selection and
   misleader-generator modules. The paper (arXiv 2601.12983v4) does not print the reader prompt or the reader decoding
   settings. We therefore re-implemented the QA evaluation (the plan's documented fallback) on the *released rendered
   charts* from the AttackViz parquet:
   - prompt: "Answer the question using the chart. Reply with only the final answer (a single number or a short phrase), without units or explanation. Question: {q}"
   - greedy decoding, max 32 new tokens, bf16, HF Transformers `AutoModelForImageTextToText`
   - relaxed accuracy: numeric answers within 5% of gold (the paper's definition); strings are matched case-insensitively
     (exact, or gold contained in a short prediction)
   - deception rate (orig. correct): among (question, misleading chart) pairs whose clean chart was answered correctly,
     the share answered with the attacker's `misleading_gold` (same relaxed matching)
2. **Subset.** N = 500 questions stratified by chart type (167 h_bar, 167 line, 166 v_bar; seed 0;
   `runs/A/subset_ids.txt`). Each question is evaluated on its correct chart and on *all* of its misleading variants
   (870 misleading charts), i.e. 1,370 images per reader.
3. **Readers.** Qwen3-VL-2B-Instruct and InternVL-3.5-2B (HF port), both in the original study's model list. Qwen3-VL-4B
   and Ovis-2.5-2B are omitted: 4B does not fit 6 GB VRAM in bf16, and Ovis needs custom remote code. Exact checkpoints
   are in `configs/models.yaml`.
4. **Reported per-model values are not recoverable.** The paper gives per-model accuracies only as plotted markers
   (Fig. 4). Comparison is therefore against the numbers stated in the text: average in-domain drop 17.2 pp
   (range 4.4-26.6 pp), mid-range group (incl. Qwen3-VL and smaller InternVL variants) 52-77% clean accuracy with
   12-21 pp drops, deception rate 11.2% (orig. correct) / 1.7% (orig. incorrect). See `runs/A/reported.csv`.
5. **Tier 2 (guard).** Not yet run: it needs the 7B code LLM attacker (Qwen2.5-Coder-7B) with the fine-tuned
   demonstration retriever; it does not fit this GPU's budget. To be re-planned with OpenRouter-hosted attackers.
