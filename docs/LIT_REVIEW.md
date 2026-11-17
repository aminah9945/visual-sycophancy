# Literature Review Protocol: Visual Sycophancy

This is the procedure for a systematic, reproducible literature review, plus the commands to run and manage it. The goal is not "read many papers". The goals are to:

1. **Prove the gap.** Show that nobody has already measured hypothesis-induced chart distortion by LLMs, or mitigated it.
2. **Build Related Work.** Write Section 2 of the paper, organised by approach rather than paper-by-paper.
3. **Verify every reference.** The brief says fabricated or unverifiable references are unacceptable.

Everything lives in `litreview/`. The single tool is `litreview/lit.py`. It searches arXiv, Semantic Scholar and OpenAlex, removes duplicates, tracks screening status, prints PRISMA counts, verifies references and exports BibTeX.

---

## 0. One-time setup

From the repository root:

```bash
python -m venv .venv
```

```bash
# Windows:  .venv\Scripts\activate
source .venv/bin/activate
```

```bash
pip install requests
```

Optional, but it avoids rate limits. A free Semantic Scholar key is available at https://www.semanticscholar.org/product/api.

```bash
# Windows PowerShell:  $env:S2_API_KEY="your-key"
export S2_API_KEY="your-key"
```

Use a reference manager as well. Install **Zotero** with the browser connector, and create one collection called `visual-sycophancy` with sub-collections `A-core-gap`, `B-misleading-vis`, `C-LLM-vis`, `D-sycophancy`, `E-VA-bias` and `baseline`.

---

## 1. Review protocol (write this down before searching)

| Item | Decision |
|---|---|
| Review questions | RQ-L1: Has any work measured whether a *stated user hypothesis* changes LLM chart design or data selection? RQ-L2: Which defences against misleading LLM-generated charts exist, and how are they evaluated? RQ-L3: Which metrics quantify misleading charts and cherry-picking? |
| Sources | arXiv, Semantic Scholar, OpenAlex (automatic); IEEE Xplore, ACM DL, ACL Anthology, Eurographics DL, Google Scholar (manual) |
| Time window | 2019–2026 for LLM work; no limit for seminal visualization work (Tufte, Pandey 2015, Shneiderman 1996) |
| Include | Peer-reviewed or arXiv; English; addresses LLM/MLLM + charts, misleading visualization, sycophancy/framing, or bias in VA |
| Exclude | Pure computer vision chart parsing with no reasoning or misleading aspect; non-English; no full text; blog posts (allowed only as motivation, never as evidence) |
| Screening | Two passes (title → abstract), then full text. Record a reason for every exclusion |
| Quality | Full-text papers are rated: venue (peer-reviewed?), evaluation (benchmark/user study?), code/data released? |

---

## 2. Automatic search

```bash
python litreview/lit.py run-queries --since 2019 --limit 50
```

This runs every line of `litreview/queries.txt` against all three sources. Results are merged and deduplicated into `litreview/data/candidates.csv`.

To add one ad-hoc query:

```bash
python litreview/lit.py search "LLM visualization hypothesis framing" --since 2022
```

**HTTP 429 messages:** these are rate limits. The script waits and retries. Set `S2_API_KEY` or run again later.

**Commit after each search session:**

```bash
git add litreview && git commit -m "lit: search round 1"
```

---

## 3. Snowballing (forward and backward citations)

Snowballing is the most reliable way to find work your keywords miss. Run it for every seed paper:

```bash
python litreview/lit.py snowball arXiv:2601.12983 --direction citations
```

```bash
python litreview/lit.py snowball arXiv:2601.12983 --direction references
```

```bash
python litreview/lit.py snowball DOI:10.1111/cgf.14559 --direction citations
```

```bash
python litreview/lit.py snowball arXiv:2310.13548 --direction citations
```

```bash
python litreview/lit.py snowball DOI:10.1109/TVCG.2021.3114804 --direction citations
```

```bash
python litreview/lit.py snowball DOI:10.1109/VAST.2017.8585669 --direction citations
```

| Seed | Why |
|---|---|
| ChartAttack (arXiv:2601.12983) | Closest work. **Anything citing it that uses benign prompts threatens your novelty.** |
| Lo et al. 2022 (10.1111/cgf.14559) | The misleader taxonomy |
| Sharma et al. 2023 (arXiv:2310.13548) | Sycophancy |
| VizLinter (10.1109/TVCG.2021.3114804) | Rule-based detection and repair |
| Wall et al. 2017 (10.1109/VAST.2017.8585669) | Bias in VA |

Repeat snowballing for each newly *included* paper that is highly relevant (one level deep is enough).

---

## 4. Manual database searches

Run these in each database's advanced search. Record each one in `litreview/data/search_log.csv`; the table below is the template.

| Database | Search string |
|---|---|
| IEEE Xplore | `("large language model" OR LLM OR GPT) AND (chart OR visualization OR visualisation) AND (misleading OR deceptive OR bias OR sycophan*)` |
| ACM DL | Same as IEEE Xplore, filtered to CHI, IUI and TOCHI, 2022–2026 |
| ACL Anthology (Google: `site:aclanthology.org`) | `chart sycophancy`, `chart misleading LLM`, `visualization generation hallucination` |
| Eurographics DL | `misleading`, `LLM`, `natural language visualization` |
| Google Scholar | `"visual sycophancy"`; `"sycophancy" "chart"`; `"misleading" "chart generation" LLM`; `"cherry-picking" visualization LLM` |

`search_log.csv` columns:

```
date,database,search_string,filters,hits,exported
2026-11-18,IEEE Xplore,"(""large language model"" OR LLM) AND ...",2022-2026,37,yes
```

For papers found manually, add a row to `candidates.csv` by hand. Alternatively, run `lit.py search "<exact title>"` so it gets metadata automatically.

---

## 5. Screening (managing `candidates.csv`)

Open `litreview/data/candidates.csv` in Excel, LibreOffice or Google Sheets. Fill in only the `status`, `reason` and `notes` columns.

| status | When |
|---|---|
| *(empty)* | Not yet screened |
| `exclude-title` | Clearly off-topic from the title |
| `exclude-abstract` | Off-topic after reading the abstract |
| `maybe` | Needs the full text |
| `exclude-fulltext` | Read it; not relevant enough |
| `include` | Will be cited or used |
| `duplicate` | Same work as another row (e.g., arXiv + published version); keep the published one |

Rules:
- Always write a `reason` for an exclusion (e.g., "no LLM", "vision-only parsing", "not charts").
- Process the whole file title-first, then do abstracts. Don't mix passes.
- **Red-flag rule:** if any paper studies *benign/neutral users* and *LLM chart generation* and *distortion or misleaders*, set `notes = THREAT` and read it the same day.

Check progress at any time:

```bash
python litreview/lit.py stats
```

This prints PRISMA-style counts. Put them in the paper as one sentence or a small flow figure.

---

## 6. Full-text reading and data extraction

For every `include` paper, add a row to `litreview/data/extraction.csv`:

```
key,citation_key,theme,problem,method,data,models,metrics,main_result,limitations,relation_to_our_gap,quality
```

- `theme` is one of A–E from `queries.txt`.
- `relation_to_our_gap` is the important column. Write one sentence on what this paper does **not** do that you do (e.g., "malicious prompter only; data unchanged").
- `quality`: peer-reviewed (Y/N), evaluation type, code released (Y/N).

Save the PDFs into Zotero rather than the repository.

---

## 7. Novelty check (do this before 5 Dec and again before any submission)

1. Filter `candidates.csv` to `notes = THREAT`. For each one, answer: same threat model (benign user)? Same output (generated charts)? Same measurement (data-level selection)? Same defence?
2. Run the "kill queries", which are designed to *find* a paper that already did your work:
   ```bash
   python litreview/lit.py search "sycophancy chart generation" --since 2025
   ```
   ```bash
   python litreview/lit.py search "user belief LLM visualization design" --since 2025
   ```
   ```bash
   python litreview/lit.py search "benign prompt misleading chart LLM" --since 2025
   ```
   ```bash
   python litreview/lit.py search "hypothesis framing chart generation" --since 2025
   ```
3. Re-run ChartAttack forward snowballing (Section 3, first command); new citing papers appear over time.
4. **If a close match exists, don't abandon the topic.** Reposition: cite it, add it to Table I, and narrow your claim to what it lacks (e.g., data-grounded defence, cherry-picking metric, VA interface).

---

## 8. Reference verification and BibTeX

Fetch BibTeX for everything marked `include`:

```bash
python litreview/lit.py bibtex --out litreview/data/included.bib
```

Verify every `\bibitem` in the paper (DOI or arXiv lookup, or a Crossref title match):

```bash
python litreview/lit.py verify paper/main.tex
```

| Output | Meaning and action |
|---|---|
| `OK` | DOI or arXiv id resolves and the title matches |
| `NO-DOI` | A confident Crossref match was found. Add the printed DOI |
| `CHECK` | The match is a preprint, reprint or different year. Find the original publication's DOI by hand |
| `FAIL` | The DOI doesn't exist. **Fix immediately** |
| `UNVERIFIED` | Book or unusual format. Check by hand against the publisher page |
| `MISSING` / `UNCITED` | A citation has no reference entry, or a reference is never cited. The brief forbids both |

Run `verify` before every submission. The target is zero `FAIL`, `MISSING` and `UNCITED` entries, and every `UNVERIFIED` entry checked by hand.

---

## 9. Turning the review into Section 2 (Related Work)

Organise by approach, not paper-by-paper (the brief requires this):

| Subsection | Themes | Synthesis sentence to end it with |
|---|---|---|
| Misleading visualizations and their detection | B | Linters and detectors judge encodings without knowing the user's claim |
| LLMs generating charts | C + ChartAttack | Generation-side risk studied only for malicious prompters, with data held fixed |
| Sycophancy and self-correction | D | Measured only in text; self-feedback may inherit the bias |
| Bias in visual analytics | E | Human treated as the bias source, not the AI assistant |

Then update **Table I** (the positioning table) with every paper flagged `THREAT` or highly relevant. Columns: generates charts / benign user / data-subset selection / data-grounded defence. Add the memo papers (Balani & Panda 2026, Wang et al. 2026, Bertrán et al. 2026, Nadib et al. 2026, Hong & Crisan 2025, Seo et al. 2026, and others) **only after** they appear as `OK` in `verify`.

---

## 10. Schedule

| Date | Task |
|---|---|
| 18 Nov | Run `run-queries` and all snowball commands; manual searches in IEEE Xplore and Google Scholar |
| 19–21 Nov | Title and abstract screening (about 1 h/day alongside experiments) |
| 22–25 Nov | Full-text reading of `maybe` and `include` papers; fill `extraction.csv` |
| 26 Nov | Novelty check (Section 7); update Table I |
| 1–2 Dec | Rewrite Section 2 from `extraction.csv` |
| 4 Dec | Final `verify`, kill queries again, `stats` for the PRISMA sentence |

Commit after each session:

```bash
git add litreview docs && git commit -m "lit: screening pass 1" && git push
```
