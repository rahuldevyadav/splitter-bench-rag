
# splitter-bench-rag

Benchmark **LangChain text splitters** end-to-end for real-world RAG:

- **Chunking** → **Embeddings** → **pgvector/Postgres** indexing → **Retrieval** → **Answer** → **LLM Judge**
- **Track A (Retrieval):** Hit@K, MRR@K, latency
- **Track B (RAG Quality):** LLM-graded answer quality (0–5 rubric)
- **Extras:**
  1. Real evaluation query generation per corpus
  2. One radar chart per corpus (automated plots)
  3. Failure gallery (bad chunks → bad answers)
  4. Cost per 1k queries per splitter

---

## Corpora (your use cases)

This repo integrates and benchmarks across these sources:

1. **GDPR (EUR-Lex PDF)**  
   `https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679`

2. **LangChain docs site (sitemap crawl)**  
   `https://docs.langchain.com/sitemap.xml` (filtered to OSS Python pages)

3. **LangChain GitHub repo Markdown**  
   `langchain-ai/langchain` (loads `.md` files via GitHub API)

Each document retains rich metadata (source URL, page number, etc.) so retrieval scoring can check “gold” matches.

---

## What you get (outputs)

After a full run you’ll have:

- `results/results.csv` — row-per-(corpus_id, splitter_config, query_id)
- `results/RESULTS.md` — GitHub-friendly leaderboard + per-corpus tables
- `results/plots/radar_*.png` — radar charts per corpus per top splitters
- `results/FAILURES.md` — failure gallery with retrieved chunks + judge rationale

---

## Prerequisites

- Python **3.10+**
- Docker
- OpenAI API key
- GitHub Personal Access Token (needed for GitHub loader)

---

## Setup

### 1) Start Postgres + pgvector

```bash
docker compose up -d
````

### 2) Install Python deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 3) Configure environment

```bash
cp .env.example .env
# Fill OPENAI_API_KEY and GITHUB_PERSONAL_ACCESS_TOKEN
```

Required `.env` fields:

* `OPENAI_API_KEY`
* `GITHUB_PERSONAL_ACCESS_TOKEN`
* (optional) `OPENAI_QGEN_MODEL`, `OPENAI_ANSWER_MODEL`, `OPENAI_JUDGE_MODEL` (defaults to `gpt-4.1-mini`)
* Postgres vars: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`

Note: `gpt-5-*` models require a verified organization; set the model env vars only if your org has access.

---

## Run the full pipeline

### Step A — Ingest corpora (download/crawl/load → processed JSONL)

```bash
python -m splitter_bench.pipelines.ingest data/corpora_manifest.yaml
```

Writes:

* `data/processed/gdpr_eurlex_pdf.jsonl`
* `data/processed/langchain_docs_site.jsonl`
* `data/processed/langchain_github_md.jsonl`

---

### Step B — Generate evaluation queries (real evalsets)

Edit `data/evalsets/manifest.json` to set evalset sizes, then run:

```bash
python scripts/build_evalset_with_llm.py
```

Writes:

* `data/evalsets/gdpr_eurlex_pdf.jsonl`
* `data/evalsets/langchain_docs_site.jsonl`
* `data/evalsets/langchain_github_md.jsonl`

Each eval item includes:

* `question` and reference `answer`
* `gold.quote` (verbatim quote from passage)
* `gold.metadata_match` (keys used to score retrieval)

> This makes retrieval evaluation “grounded”: the gold target is explicitly tied to a real passage + metadata.

---

### Step C — Index (splitters → chunks → embeddings → pgvector)

```bash
python -m splitter_bench.pipelines.index --all-splitters
```

Creates one PGVector collection per (corpus, splitter_config), e.g.:

* `vs_gdpr_eurlex_pdf__<splitter_key>`
* `vs_langchain_docs_site__<splitter_key>`
* `vs_langchain_github_md__<splitter_key>`

---

### Step D — Benchmark (Track A + Track B + costs)

```bash
python -m splitter_bench.pipelines.benchmark --all-splitters --out results/results.csv
```

This:

* Retrieves top-K chunks per query
* Scores Track A (Hit@K, MRR@K, latency)
* Generates an answer using retrieved context
* Judges answer quality (0–5 + rationale)
* Records token usage + computes cost per 1k queries

---

### Step E — Render GitHub-ready reports

```bash
python scripts/render_results_md.py
python scripts/plot_radar_per_corpus.py
python scripts/build_failure_gallery.py
```

Outputs:

* `results/RESULTS.md`
* `results/plots/radar_*.png`
* `results/FAILURES.md`

---

## Interpreting metrics

### Track A (Retrieval)

* **Hit@K**: did we retrieve a chunk matching `gold.metadata_match` in the top K?
* **MRR@K**: mean reciprocal rank (higher is better)
* **Latency**: recorded per query (also reported at P95 in RESULTS)

### Track B (RAG answer quality)

* **Judge score (0–5)**:

  * 5 = fully correct, supported by retrieved context
  * 3 = partially correct / weakly supported
  * 1 = mostly incorrect / hallucinated
  * 0 = irrelevant/unsafe
* `% ≥ 4` is a helpful “good answer rate”

### Cost per 1k queries

Per-query cost is estimated from:

* query embedding token estimate (tiktoken)
* answer model usage (if provided by API)
* judge model usage (if provided by API)

`cost_per_1k_queries_usd = per_query_cost * 1000`

---

## Adding / tuning splitters

Edit:

* `src/splitter_bench/splitters/configs.yaml`

Example:

```yaml
splitters:
  - name: token_text
    type: token
    params:
      chunk_size: [512, 800, 1200]
      chunk_overlap: [64, 128]
      encoding_name: ["cl100k_base"]
```

Then rerun **index** and **benchmark**.

---

## Failure gallery (why it’s useful)

`results/FAILURES.md` shows cases where:

* Retrieval succeeded (**Hit@K=1**)
* But the judged answer was poor (**<=2**)

It highlights boundary pathologies like:

* hyphenation splits
* unclosed code fences
* markdown table fragmentation
* GDPR “Article X” header fragmentation

Use it to pick better splitters/configs per corpus.

---

## What to commit to GitHub

Safe to commit:

* code
* `data/corpora_manifest.yaml`
* `data/evalsets/manifest.json`
* generated `results/RESULTS.md`, `results/FAILURES.md`, `results/plots/*`

Avoid committing:

* large raw crawls or PDFs if licensing is unclear
* private or sensitive data

---

## Troubleshooting

### GitHub loader fails

* Ensure `GITHUB_PERSONAL_ACCESS_TOKEN` is set in `.env`

### PDF parsing fails

* Install PyMuPDF:

  ```bash
  pip install pymupdf
  ```

### No evalsets found

* Run:

  ```bash
  python scripts/build_evalset_with_llm.py
  ```

### Postgres connection errors

* Verify docker is running:

  ```bash
  docker ps
  ```
* Check `.env` PG vars match docker-compose.

---

## License / ethics

* GDPR is public legal text from EUR-Lex.
* LangChain docs + GitHub repo are public; respect source terms and rate limits.
* Don’t ingest or publish private/PII content.

---

```
::contentReference[oaicite:0]{index=0}
```
