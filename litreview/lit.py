#!/usr/bin/env python3
"""Literature-review helper for the Visual Sycophancy project.

Subcommands
  search       one query against arXiv / Semantic Scholar / OpenAlex -> candidates.csv
  run-queries  every query in queries.txt
  snowball     papers citing (or cited by) a seed paper, via Semantic Scholar
  stats        PRISMA-style counts from the 'status' column of candidates.csv
  verify       check every \\bibitem in a .tex file against Crossref / arXiv
  bibtex       fetch BibTeX for all rows with status=include (needs a DOI or arXiv id)

Only dependency: requests.  Optional env vars: S2_API_KEY, OPENALEX_EMAIL.
"""
import argparse
import csv
import difflib
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CAND = DATA / "candidates.csv"
FIELDS = ["key", "title", "authors", "year", "venue", "doi", "arxiv", "url",
          "citations", "source", "query", "status", "reason", "notes", "abstract"]
UA = {"User-Agent": "lit-review-script/1.0"}
VALID_STATUS = {"", "include", "maybe", "exclude-title", "exclude-abstract",
                "exclude-fulltext", "duplicate"}


# ---------------------------------------------------------------- utilities
def get(url, params=None, headers=None, tries=6):
    h = dict(UA, **(headers or {}))
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=30)
        except requests.RequestException as e:
            print(f"  network error {e}; retry", file=sys.stderr)
            time.sleep(2 ** i)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(60, 2 ** (i + 1))
            print(f"  HTTP {r.status_code} from {url.split('/')[2]}; waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        return r
    return None


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def load():
    if not CAND.exists():
        return []
    with CAND.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows):
    DATA.mkdir(parents=True, exist_ok=True)
    with CAND.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge(rows, new):
    """Deduplicate on DOI, arXiv id, then normalised title."""
    idx = {}
    for r in rows:
        for k in (r.get("doi", "").lower(), r.get("arxiv", ""), norm_title(r.get("title"))):
            if k:
                idx[k] = r
    added = 0
    for n in new:
        keys = [n.get("doi", "").lower(), n.get("arxiv", ""), norm_title(n.get("title"))]
        hit = next((idx[k] for k in keys if k and k in idx), None)
        if hit:
            for f in ("doi", "arxiv", "venue", "citations", "abstract"):
                if not hit.get(f) and n.get(f):
                    hit[f] = n[f]
            if n["source"] not in hit["source"]:
                hit["source"] += ";" + n["source"]
            continue
        n.setdefault("status", "")
        n["key"] = f"P{len(rows) + 1:04d}"
        rows.append(n)
        for k in keys:
            if k:
                idx[k] = n
        added += 1
    return added


# ---------------------------------------------------------------- sources
def arxiv_search(q, since, limit):
    r = get("https://export.arxiv.org/api/query",
            {"search_query": f"all:{q}", "max_results": limit,
             "sortBy": "relevance"})
    if r is None or r.status_code != 200:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        year = int(e.find("a:published", ns).text[:4])
        if year < since:
            continue
        aid = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        doi = e.find("{http://arxiv.org/schemas/atom}doi")
        out.append(dict(
            title=" ".join(e.find("a:title", ns).text.split()),
            authors="; ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns)),
            year=year, venue="arXiv", doi=doi.text if doi is not None else "",
            arxiv=aid, url=f"https://arxiv.org/abs/{aid}", citations="",
            abstract=" ".join(e.find("a:summary", ns).text.split())[:1500],
            source="arxiv", query=q))
    time.sleep(3)  # arXiv asks for >= 3 s between calls
    return out


def s2_row(p, q, src):
    ext = p.get("externalIds") or {}
    return dict(
        title=p.get("title") or "", year=p.get("year") or "",
        authors="; ".join(a.get("name", "") for a in (p.get("authors") or [])),
        venue=p.get("venue") or "", doi=ext.get("DOI", "") or "",
        arxiv=ext.get("ArXiv", "") or "", url=p.get("url") or "",
        citations=p.get("citationCount") or "",
        abstract=(p.get("abstract") or "")[:1500], source=src, query=q)


S2_FIELDS = "title,year,authors,venue,externalIds,url,citationCount,abstract"


def s2_headers():
    k = os.environ.get("S2_API_KEY")
    return {"x-api-key": k} if k else {}


def s2_search(q, since, limit):
    r = get("https://api.semanticscholar.org/graph/v1/paper/search",
            {"query": q, "limit": min(limit, 100), "fields": S2_FIELDS,
             "year": f"{since}-"}, s2_headers())
    if r is None or r.status_code != 200:
        return []
    time.sleep(1.1)
    return [s2_row(p, q, "s2") for p in r.json().get("data", [])]


def openalex_search(q, since, limit):
    params = {"search": q, "per-page": min(limit, 200),
              "filter": f"from_publication_date:{since}-01-01"}
    if os.environ.get("OPENALEX_EMAIL"):
        params["mailto"] = os.environ["OPENALEX_EMAIL"]
    r = get("https://api.openalex.org/works", params)
    if r is None or r.status_code != 200:
        return []
    out = []
    for w in r.json().get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        inv = w.get("abstract_inverted_index") or {}
        words = sorted((p, t) for t, ps in inv.items() for p in ps)
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        out.append(dict(
            title=w.get("title") or "", year=w.get("publication_year") or "",
            authors="; ".join(a["author"]["display_name"] for a in w.get("authorships", [])),
            venue=loc.get("display_name") or "", doi=doi, arxiv="",
            url=w.get("id"), citations=w.get("cited_by_count") or "",
            abstract=" ".join(t for _, t in words)[:1500],
            source="openalex", query=q))
    return out


SOURCES = {"arxiv": arxiv_search, "s2": s2_search, "openalex": openalex_search}


# ---------------------------------------------------------------- commands
def do_search(queries, sources, since, limit):
    rows = load()
    for q in queries:
        for s in sources:
            res = SOURCES[s](q, since, limit)
            n = merge(rows, res)
            print(f"[{s:8}] {len(res):3} hits, {n:3} new  <- {q}")
    save(rows)
    print(f"candidates.csv now has {len(rows)} unique papers")


def cmd_search(a):
    do_search([a.query], a.sources.split(","), a.since, a.limit)


def cmd_run_queries(a):
    qs = [l.strip() for l in Path(a.file).read_text().splitlines()
          if l.strip() and not l.startswith("#")]
    do_search(qs, a.sources.split(","), a.since, a.limit)


def cmd_snowball(a):
    pid = a.paper
    url = f"https://api.semanticscholar.org/graph/v1/paper/{pid}/{a.direction}"
    r = get(url, {"fields": S2_FIELDS, "limit": 1000}, s2_headers())
    if r is None or r.status_code != 200:
        sys.exit(f"Semantic Scholar error: {r.status_code if r else 'no response'}")
    key = "citingPaper" if a.direction == "citations" else "citedPaper"
    res = [s2_row(d[key], f"snowball:{a.direction}:{pid}", "s2-snowball")
           for d in r.json().get("data", []) if d.get(key, {}).get("title")]
    rows = load()
    n = merge(rows, res)
    save(rows)
    print(f"{len(res)} papers in {a.direction} of {pid}; {n} new")


def cmd_stats(a):
    rows = load()
    bad = [r["key"] for r in rows if r.get("status", "") not in VALID_STATUS]
    c = {}
    for r in rows:
        c[r.get("status", "")] = c.get(r.get("status", ""), 0) + 1
    tot = len(rows)
    print("PRISMA-style flow")
    print(f"  records identified (after automatic dedup): {tot}")
    print(f"  manual duplicates removed:                  {c.get('duplicate', 0)}")
    scr = tot - c.get("duplicate", 0)
    print(f"  records screened (title):                   {scr}")
    print(f"  excluded on title:                          {c.get('exclude-title', 0)}")
    print(f"  excluded on abstract:                       {c.get('exclude-abstract', 0)}")
    print(f"  full texts assessed:                        "
          f"{c.get('include', 0) + c.get('exclude-fulltext', 0) + c.get('maybe', 0)}")
    print(f"  excluded on full text:                      {c.get('exclude-fulltext', 0)}")
    print(f"  still 'maybe':                              {c.get('maybe', 0)}")
    print(f"  INCLUDED:                                   {c.get('include', 0)}")
    print(f"  not yet screened:                           {c.get('', 0)}")
    if bad:
        print(f"WARNING invalid status values in: {', '.join(bad[:20])}")


def crossref_doi(doi):
    r = get(f"https://api.crossref.org/works/{doi}")
    if r is None or r.status_code != 200:
        return None
    m = r.json()["message"]
    return dict(title=(m.get("title") or [""])[0],
                year=(m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                venue=(m.get("container-title") or [""])[0],
                authors=[f"{x.get('given', '')} {x.get('family', '')}".strip()
                         for x in m.get("author", [])])


def crossref_title(title):
    r = get("https://api.crossref.org/works",
            {"query.bibliographic": title, "rows": 1})
    if r is None or r.status_code != 200:
        return None
    items = r.json()["message"]["items"]
    if not items:
        return None
    m = items[0]
    return dict(title=(m.get("title") or [""])[0], doi=m.get("DOI"),
                year=(m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                venue=(m.get("container-title") or [""])[0])


def arxiv_id(aid):
    r = get("https://export.arxiv.org/api/query", {"id_list": aid})
    if r is None or r.status_code != 200:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = ET.fromstring(r.text).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return None
    return dict(title=" ".join(e.find("a:title", ns).text.split()),
                year=e.find("a:published", ns).text[:4])


def cmd_verify(a):
    tex = Path(a.tex).read_text(encoding="utf-8")
    items = re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", tex, re.S)
    cited = set(k.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", tex) for k in grp.split(","))
    print(f"{len(items)} bibitems, {len(cited)} distinct cited keys\n")
    for key in sorted(cited - {k for k, _ in items}):
        print(f"MISSING  cited but no bibitem: {key}")
    for key, body in items:
        if key not in cited:
            print(f"UNCITED  {key}")
        flat = " ".join(body.split())
        m = re.search(r"``(.+?),?''", flat)
        title = re.sub(r"[{}\\]", "", m.group(1)) if m else ""
        doi = re.search(r"doi:\s*(10\.\S+?)[.,]?(\s|$)", flat)
        arx = re.search(r"arXiv:(\d{4}\.\d{4,5})", flat)
        if doi:
            got = crossref_doi(doi.group(1))
            if not got:
                print(f"FAIL     {key}: DOI {doi.group(1)} not found in Crossref")
                continue
            s = sim(title, got["title"]) if title else 1
            tag = "OK      " if s > 0.85 else "CHECK   "
            print(f"{tag} {key}: DOI resolves -> '{got['title']}' ({got['venue']}, {got['year']}) sim={s:.2f}")
        elif arx:
            got = arxiv_id(arx.group(1))
            if not got:
                print(f"FAIL     {key}: arXiv {arx.group(1)} not found")
                continue
            s = sim(title, got["title"]) if title else 1
            tag = "OK      " if s > 0.85 else "CHECK   "
            print(f"{tag} {key}: arXiv -> '{got['title']}' ({got['year']}) sim={s:.2f}")
        elif title:
            got = crossref_title(title)
            years = re.findall(r"\b(19\d\d|20\d\d)\b", flat)
            if got and sim(title, got["title"]) > 0.85:
                suspicious = (not got["venue"]) or (years and str(got["year"]) != years[-1])
                tag = "CHECK   " if suspicious else "NO-DOI  "
                hint = " (venue/year differ: may be a preprint or reprint - find the original DOI)" if suspicious else " -> add it"
                print(f"{tag} {key}: Crossref match DOI {got['doi']} ({got['venue']}, {got['year']}){hint}")
            else:
                print(f"UNVERIFIED {key}: no DOI/arXiv and no confident Crossref match - check by hand")
        else:
            print(f"UNVERIFIED {key}: could not parse a title (book or unusual format) - check by hand")
        time.sleep(0.5)


def cmd_bibtex(a):
    rows = [r for r in load() if r.get("status") == "include"]
    out = []
    for r in rows:
        bib = None
        if r.get("doi"):
            x = get(f"https://doi.org/{r['doi']}", headers={"Accept": "application/x-bibtex"})
            if x is not None and x.status_code == 200:
                bib = x.text.strip()
        if not bib and r.get("arxiv"):
            x = get(f"https://arxiv.org/bibtex/{r['arxiv']}")
            if x is not None and x.status_code == 200:
                bib = x.text.strip()
        if bib:
            out.append(bib)
        else:
            print(f"no BibTeX for {r['key']} {r['title'][:60]}")
    Path(a.out).write_text("\n\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} entries to {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("search"); s.add_argument("query")
    s.add_argument("--sources", default="arxiv,s2,openalex")
    s.add_argument("--since", type=int, default=2022); s.add_argument("--limit", type=int, default=50)
    s.set_defaults(f=cmd_search)
    s = sp.add_parser("run-queries"); s.add_argument("--file", default=str(HERE / "queries.txt"))
    s.add_argument("--sources", default="arxiv,s2,openalex")
    s.add_argument("--since", type=int, default=2022); s.add_argument("--limit", type=int, default=50)
    s.set_defaults(f=cmd_run_queries)
    s = sp.add_parser("snowball"); s.add_argument("paper", help="e.g. arXiv:2601.12983 or DOI:10.1109/...")
    s.add_argument("--direction", choices=["citations", "references"], default="citations")
    s.set_defaults(f=cmd_snowball)
    s = sp.add_parser("stats"); s.set_defaults(f=cmd_stats)
    s = sp.add_parser("verify"); s.add_argument("tex"); s.set_defaults(f=cmd_verify)
    s = sp.add_parser("bibtex"); s.add_argument("--out", default=str(DATA / "included.bib"))
    s.set_defaults(f=cmd_bibtex)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
