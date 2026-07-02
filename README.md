# Search & Ranking Engine

A two-stage search engine over MS MARCO passages: a from-scratch BM25 retriever for candidate generation, followed by a LightGBM re-ranker trained on hand-engineered features. Served via FastAPI, containerized with Docker, and load-tested with Locust.

## Architecture

```
Query
  │
  ▼
Stage 1 — BM25Retriever (custom inverted index)
  retrieves top-100 candidate documents
  │
  ▼
Stage 2 — LightGBMRanker
  extracts 5 features per (query, doc) pair, re-scores with a trained model
  │
  ▼
Top-k results — FastAPI /search endpoint
```

### Stage 1 — Retrieval

- **`InvertedIndex`** (`src/indexer`): a hand-built inverted index — tokenizes text, builds postings lists per term, and tracks per-document length plus a corpus-wide average document length (updated incrementally as documents are added). Serializes to/from JSON for persistence across restarts.
- **`BM25Retriever`** (`src/retrieval`): implements the BM25 scoring formula directly (`k1=1.5`, `b=0.75` by default) rather than using the `rank-bm25` library. Candidate generation is OR-based — any document containing at least one query term is scored — then results are sorted by score.

### Stage 2 — Re-ranking

- **`RankerFeatures`**: extracts 5 features per (query, doc) pair — BM25 score, query length, document length, exact term-overlap count, and average term frequency of query terms in the document.
- **`LightGBMRanker`**: trains a LightGBM model on these features to re-score BM25's candidate set.

  **Note:** currently trained with `objective="binary"` (binary classification on relevance labels), not a ranking-specific objective like `lambdarank`. It functions as a pointwise reranker rather than a true learning-to-rank model — see "What I'd improve next."

### Serving

- FastAPI app (`src/serving/app.py`) exposes:
  - `POST /index` — build the inverted index from a list of documents
  - `POST /train-ranker` — train the LightGBM reranker on labeled (query, doc, relevance) triples
  - `POST /search` — run the full two-stage pipeline and return ranked results
  - `GET /health` — index status and doc/vocab counts
- Containerized with Docker; `docker-compose.yml` runs the API with a health check, plus an optional Locust load-testing profile (`docker compose --profile loadtest up`).

## Fixing train/test leakage

`build_eval_corpus.py` addresses two leakage problems found during evaluation:

1. **Gold passages didn't survive random sampling.** A random 10k-passage subsample from the ~8.8M-passage collection meant only about 26 of 100k judged queries still had their gold (relevant) passage present — a ~0.11% survival rate.
2. **Train/test query overlap.** The reranker was being trained and evaluated on the same query set, so it was effectively graded on data it had already seen.

**Fix:** queries are split into disjoint train/test sets *before* any triples or passages are built, and gold passages are explicitly guaranteed to be included in the output corpus rather than hoping they survive random sampling. Distractor passages are added via reservoir sampling to pad the corpus to a target size without bias.

```
data/eval_v2/
├── passages.tsv        # gold passages + sampled distractors
├── queries.train.tsv   # build reranker training triples from these only
├── qrels.train.tsv
├── queries.test.tsv    # held out — never used in training
└── qrels.test.tsv
```

## Results

<!-- TODO: run evaluation on the real MS MARCO eval_v2 corpus (using the same ranx-based approach as test_eval_pipeline.py) and fill in actual numbers. -->

| Metric | BM25 only | BM25 + LightGBM re-rank |
|---|---|---|
| NDCG@10 | TODO | TODO |
| MRR@10 | TODO | TODO |

Evaluated on `queries.test.tsv` / `qrels.test.tsv` from `build_eval_corpus.py` — held-out queries with no train/test overlap.

## Project structure

```
.
├── src/
│   ├── indexer/          # InvertedIndex
│   ├── retrieval/        # BM25Retriever
│   ├── reranker/         # RankerFeatures, LightGBMRanker
│   └── serving/          # FastAPI app
├── tests/
│   ├── test_indexer.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_serving.py       # integration tests via FastAPI TestClient
│   ├── test_eval_pipeline.py # ranx-based NDCG/MRR checks on synthetic corpus
│   └── locustfile.py         # load test
├── build_eval_corpus.py      # leakage-safe train/test corpus builder
├── load_index.py             # pushes a built passages.tsv to the running API's /index endpoint
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Running locally

```bash
# 1. Install
pip install -r requirements.txt

# 2. Build the leakage-safe eval corpus from raw MS MARCO files
python build_eval_corpus.py \
    --collection data/msmarco/collection.tsv \
    --queries data/msmarco/queries.dev.small.tsv \
    --qrels data/msmarco/qrels.dev.small.tsv \
    --out-dir data/eval_v2

# 3. Start the API
docker compose up

# 4. Load the corpus into the running index
python load_index.py

# 5. Query it
curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"query": "how does photosynthesis work", "top_k": 10}'
```

## Testing

```bash
pytest tests/
```

Covers: inverted index correctness, BM25 scoring properties (term frequency, IDF weighting, multi-term queries), reranker training/inference, full API integration (index → train → search), and NDCG/MRR sanity checks on a synthetic corpus via `ranx`.

## Load testing

```bash
docker compose --profile loadtest up
```

Runs Locust against `/search` with realistic MS MARCO-style queries (10:1 weighted toward search vs. health checks), spins up at `http://localhost:8089`.

## What I'd improve next

- **Switch the reranker to a true ranking objective.** It currently trains with `objective="binary"` in LightGBM; switching to `lambdarank` (or another pairwise/listwise objective) would make this genuine learning-to-rank rather than binary classification repurposed as a score.
- **Add a Redis caching layer** for repeated-query latency — not yet implemented.
- **Add semantic retrieval.** `sentence-transformers` is already a dependency but isn't wired into the pipeline yet; a dense retrieval stage (or hybrid BM25 + embedding re-ranking) would help with vocabulary-mismatch queries that BM25 misses entirely.
- **Run and publish real eval numbers** on the full `eval_v2` corpus rather than only the synthetic smoke-test corpus used in `test_eval_pipeline.py`.

## Tech stack

Python · custom inverted index · BM25 (from scratch) · LightGBM · FastAPI · Docker · Locust · ranx (eval) · pytest
