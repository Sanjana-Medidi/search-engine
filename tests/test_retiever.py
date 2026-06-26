from src.indexer.indexer import InvertedIndex
from src.retrieval.retriever import BM25Retriever

def _build_index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.add_document(1, "the quick brown fox jumps over the lazy dog")
    idx.add_document(2, "the lazy dog slept all day")
    idx.add_document(3, "the fox ran quickly across the field")
    return idx

def test_higher_tf_ranks_higher():
    #A doc where the query term appears more often should rank higher.
    idx = InvertedIndex()
    idx.add_document(1, "fox fox fox chased the rabbit")
    idx.add_document(2, "the fox sat quietly")
    retriever = BM25Retriever(idx)

    results = retriever.retrieve("fox")
    top_doc = results[0][0]
    assert top_doc ==1, "Doc with tf=3 should outscore doc with tf=1"

def test_rare_term_scores_higher_than_common():
    #ensure rare terms carry more weight than common ones.
    idx = InvertedIndex()
    idx.add_document(1, "the the the fox")   # 'the' is in every doc
    idx.add_document(2, "the quick fox")
    idx.add_document(3, "the slow fox")
    retriever = BM25Retriever(idx)

    score_rare = retriever.score("quick", 2)
    score_common = retriever.score("the", 2)
    assert score_rare>score_common

def test_retrieve_returns_ranked_list():
    #Results must be sorted descending by score.
    idx = _build_index()
    retriever = BM25Retriever(idx)
    results = retriever.retrieve("fox")

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_unknown_query_returns_empty():
    idx = _build_index()
    retriever = BM25Retriever(idx)
    assert retriever.retrieve("unknownterm") == []

def test_score_is_positive():
    idx = _build_index()
    retriever = BM25Retriever(idx)
    s = retriever.score("fox", 1)
    assert s> 0.0

def test_multi_term_query_sums_all_terms():
    idx = InvertedIndex()
    idx.add_document(1, "fox jumps over fox")
    idx.add_document(2, "fox sleeps")
    retriever = BM25Retriever(idx)

    single = retriever.score("fox", 1)
    multi = retriever.score("fox jumps", 1)
    assert multi > single, "a two-term query should score higher than just one of its terms"
