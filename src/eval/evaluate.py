from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ranx import Qrels, Run, evaluate

from src.indexer.indexer import InvertedIndex
from src.reranker.lightgbm_ranker import LightGBMRanker
from src.retrieval.retriever import BM25Retriever


def load_eval_split(eval_dir: Path):
    """Load the leakage-free corpus + disjoint train/test query splits
    produced by build_eval_corpus.py."""
    passages: dict[int, str] = {}
    with open(eval_dir / "passages.tsv", encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            passages[int(pid)] = text

    def _load_queries(path: Path) -> dict[int, str]:
        out: dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                out[int(qid)] = text
        return out

    def _load_qrels(path: Path) -> dict[int, dict[int, int]]:
        out: dict[int, dict[int, int]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                qid, _, pid, rel = line.rstrip("\n").split("\t")
                out.setdefault(int(qid), {})[int(pid)] = int(rel)
        return out

    train_queries = _load_queries(eval_dir / "queries.train.tsv")
    train_qrels = _load_qrels(eval_dir / "qrels.train.tsv")
    test_queries = _load_queries(eval_dir / "queries.test.tsv")
    test_qrels = _load_qrels(eval_dir / "qrels.test.tsv")

    return passages, train_queries, train_qrels, test_queries, test_qrels


def build_index(passages: dict[int, str], index_cache: Path) -> InvertedIndex:
    """Build or load cached index. Cache path is now keyed off --eval-dir
    so switching corpora can't silently serve a stale index."""
    if index_cache.exists():
        print(f"Loading cached index from {index_cache}...")
        return InvertedIndex.load(index_cache)

    print(f"Building index over {len(passages):,} passages...")
    idx = InvertedIndex()
    t0 = time.perf_counter()
    for pid, text in passages.items():
        idx.add_document(pid, text)
    elapsed = time.perf_counter() - t0

    print(f"Indexed {len(passages):,} docs in {elapsed:.1f}s ({len(passages)/elapsed:,.0f} docs/s)")
    index_cache.parent.mkdir(parents=True, exist_ok=True)
    idx.save(index_cache)
    return idx


def create_training_data(
    queries: dict[int, str],
    qrels: dict[int, dict[int, int]],
    retriever: BM25Retriever,
) -> list[tuple[str, int, int]]:
    """Create training triples for the ranker. Call this ONLY with the
    train split -- never with queries that will also be evaluated."""
    triples = []
    print(f"Creating training data from {len(qrels):,} queries...")

    for i, (qid, rel_docs) in enumerate(qrels.items(), 1):
        if qid not in queries:
            continue

        query = queries[qid]
        candidates = retriever.retrieve(query, top_k=100)
        candidate_ids = [doc_id for doc_id, _ in candidates]

        for doc_id in candidate_ids[:50]:
            relevance = 1 if doc_id in rel_docs else 0
            triples.append((query, doc_id, relevance))

        if i % 1000 == 0:
            print(f"Processed {i:,} queries, {len(triples):,} triples created.")

    print(f"Final training dataset size: {len(triples):,} triples.")
    return triples


def run_retrieval(
    retriever: BM25Retriever,
    ranker: LightGBMRanker | None,
    queries: dict[int, str],
    qrels: dict[int, dict[int, int]],
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Execute retrieval and optional reranking. Call this with the TEST
    split for both the baseline and reranker runs, so the comparison is
    apples-to-apples and neither system has seen these queries' labels."""
    run_dict: dict[str, dict[str, float]] = {}
    eval_qids = list(qrels.keys())

    system = "BM25 + Reranker" if ranker else "BM25"
    print(f"\nExecuting {system} retrieval for {len(eval_qids):,} queries...")
    t0 = time.perf_counter()

    for i, qid in enumerate(eval_qids, 1):
        query = queries.get(qid)
        if not query:
            continue

        bm25_results = retriever.retrieve(query, top_k=100)
        candidate_ids = [doc_id for doc_id, _ in bm25_results]

        if ranker and candidate_ids:
            try:
                reranked = ranker.rerank(query, candidate_ids, top_k=top_k)
                run_dict[str(qid)] = {str(pid): float(score) for pid, score in reranked}
            except Exception:
                run_dict[str(qid)] = {str(pid): float(score) for pid, score in bm25_results[:top_k]}
        else:
            run_dict[str(qid)] = {str(pid): float(score) for pid, score in bm25_results[:top_k]}

        if i % 500 == 0:
            print(f"Progress: {i:,}/{len(eval_qids):,} ({time.perf_counter() - t0:.1f}s elapsed)")

    print(f"Retrieval complete in {time.perf_counter() - t0:.1f}s.")
    return run_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IR Pipeline (leakage-free)")
    parser.add_argument("--eval-dir", type=str, default="data/eval_v2",
                         help="Output directory from build_eval_corpus.py")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    index_cache = eval_dir / "index.json"  # keyed to this corpus, won't collide with old runs

    passages, train_queries, train_qrels, test_queries, test_qrels = load_eval_split(eval_dir)
    print(f"Dataset loaded: {len(passages):,} passages, "
          f"{len(train_queries):,} train queries, {len(test_queries):,} test queries.")

    idx = build_index(passages, index_cache)
    retriever = BM25Retriever(idx)

    # --- Baseline: BM25 on TEST queries only ---
    bm25_run_dict = run_retrieval(retriever, None, test_queries, test_qrels, args.top_k)
    qrels_ranx = Qrels({str(qid): {str(pid): rel for pid, rel in pmap.items()}
                       for qid, pmap in test_qrels.items()})

    metrics = ["ndcg@10", "mrr@10", f"recall@{args.top_k}"]
    bm25_results = evaluate(qrels_ranx, Run(bm25_run_dict, name="bm25"), metrics)

    print("\nBaseline Results (BM25):")
    for m, v in bm25_results.items():
        print(f"{m:15}: {v:.4f}")

    # --- Train reranker on TRAIN queries only ---
    training_triples = create_training_data(train_queries, train_qrels, retriever)
    ranker = LightGBMRanker(idx, retriever)
    ranker.train(training_triples)

    # --- Evaluate reranker on the SAME held-out TEST queries as the baseline ---
    reranker_run_dict = run_retrieval(retriever, ranker, test_queries, test_qrels, args.top_k)
    reranker_results = evaluate(qrels_ranx, Run(reranker_run_dict, name="bm25+reranker"), metrics)

    print("\nImproved Results (BM25 + Reranker):")
    for m, v in reranker_results.items():
        print(f"{m:15}: {v:.4f}")

    print("\nPerformance Comparison:")
    print(f"{'Metric':<15} {'BM25':<10} {'Reranker':<10} {'Delta':<10}")
    for m in metrics:
        delta = reranker_results[m] - bm25_results[m]
        print(f"{m:<15} {bm25_results[m]:.4f}     {reranker_results[m]:.4f}     {delta:+.4f}")


if __name__ == "__main__":
    main()