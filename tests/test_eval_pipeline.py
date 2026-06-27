"""
Smoke-test the end-to-end evaluation pipeline with a synthetic corpus.
Ensures core retrieval and evaluation logic remains stable without external data.
"""

from ranx import Qrels, Run, evaluate

from src.indexer.indexer import InvertedIndex
from src.retrieval.retriever import BM25Retriever


def _make_fixtures() -> tuple[dict[int, str], dict[int, str], dict[int, dict[int, int]]]:
    """Provides a small synthetic corpus for testing."""
    passages = {
        1: "python machine learning tutorial",
        2: "python web scraping guide",
        3: "javascript frontend framework",
        4: "machine learning deep learning neural network",
        5: "python data science pandas numpy",
    }
    queries = {
        101: "python machine learning",
        102: "javascript framework",
    }
    # Relevance judgments: pid is 1 if relevant, 0 otherwise
    qrels = {
        101: {1: 1, 4: 1, 5: 1},
        102: {3: 1},
    }
    return passages, queries, qrels


def _build_retriever(passages: dict[int, str]) -> BM25Retriever:
    """Builds a temporary index from the provided passages."""
    idx = InvertedIndex()
    for pid, text in passages.items():
        idx.add_document(pid, text)
    return BM25Retriever(idx)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_pipeline_produces_run_dict() -> None:
    """Verifies that the retrieval pipeline outputs correctly formatted results."""
    passages, queries, qrels = _make_fixtures()
    retriever = _build_retriever(passages)

    run_dict: dict[str, dict[str, float]] = {}
    for qid, pids in qrels.items():
        query = queries[qid]
        results = retriever.retrieve(query, top_k=10)
        run_dict[str(qid)] = {str(pid): score for pid, score in results}

    assert run_dict, "Result dictionary should not be empty."
    for scored in run_dict.values():
        assert isinstance(scored, dict)
        assert all(isinstance(s, float) for s in scored.values())


def test_relevant_passage_retrieved() -> None:
    """Checks that the retriever successfully identifies relevant documents."""
    passages, _, _ = _make_fixtures()
    retriever = _build_retriever(passages)

    results = retriever.retrieve("python machine learning", top_k=5)
    retrieved_ids = {pid for pid, _ in results}
    
    assert retrieved_ids & {1, 4, 5}, "Expected at least one relevant passage in top-5 results."


def test_ndcg_mrr_are_positive() -> None:
    """Verifies that evaluation metrics produce non-zero scores."""
    passages, queries, qrels = _make_fixtures()
    retriever = _build_retriever(passages)

    run_dict: dict[str, dict[str, float]] = {}
    for qid in qrels:
        results = retriever.retrieve(queries[qid], top_k=10)
        run_dict[str(qid)] = {str(pid): score for pid, score in results}

    qrels_ranx = Qrels({str(qid): {str(pid): rel for pid, rel in pmap.items()}
                       for qid, pmap in qrels.items()})
    run_ranx = Run(run_dict, name="bm25_smoke")
    
    metrics = evaluate(qrels_ranx, run_ranx, ["ndcg@10", "mrr@10"])

    assert metrics["ndcg@10"] > 0.0, "NDCG@10 should be positive for this corpus."
    assert metrics["mrr@10"] > 0.0, "MRR@10 should be positive for this corpus."


def test_scores_are_descending_in_run() -> None:
    """Sanity check: ensure results are sorted by relevance score."""
    passages, _, _ = _make_fixtures()
    retriever = _build_retriever(passages)

    results = retriever.retrieve("python", top_k=5)
    scores = [s for _, s in results]
    
    assert scores == sorted(scores, reverse=True), "Results must be sorted by score descending."