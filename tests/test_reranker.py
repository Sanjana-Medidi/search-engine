from src.indexer.indexer import InvertedIndex
from src.retrieval.retriever import BM25Retriever
from src.reranker.lightgbm_ranker import LightGBMRanker

def test_reranker_trains_and_ranks():
    idx = InvertedIndex()
    idx.add_document(1, "The quick brown fox jumps over the lazy dog")
    idx.add_document(2, "A fast fox leaps across the field")
    idx.add_document(3, "The dog was very lazy and slept all day")
    idx.add_document(4, "Foxes are quick animals that jump high")
    idx.add_document(5, "Dogs and foxes are natural enemies")

    retriever = BM25Retriever(idx)
    ranker = LightGBMRanker(idx, retriever)

    training_data = [
            ("quick fox", 1, 1),        # relevant: has both terms
            ("quick fox", 2, 1),        
            ("quick fox", 3, 0),        # irrelevant: no "fox"
            ("quick fox", 4, 1),        
            ("quick fox", 5, 0),        # irrelevant: has "fox" but not "quick"
            
            ("lazy dog", 1, 1),         # relevant: has both terms
            ("lazy dog", 3, 1),        
            ("lazy dog", 2, 0),         # irrelevant: has neither
            ("lazy dog", 4, 0),         
            ("lazy dog", 5, 0),  
    ]

    ranker.train(training_data)
    print("Ranker trained successfully")

    candidates_with_scores = retriever.retrieve("quick fox", top_k=100)
    candidate_ids = [doc_id for doc_id, _ in candidates_with_scores]

    reranked = ranker.rerank("quick fox", candidate_ids, top_k = 10)
    print("✓ Re-ranking successful")
    print(f"  Query: 'quick fox'")
    print(f"  BM25 top 3:")
    for doc_id, score in candidates_with_scores[:3]:
        print(f"    Doc {doc_id}: {score:.3f}")
    print(f"  After re-ranking:")
    for doc_id, score in reranked[:3]:
        print(f"    Doc {doc_id}: {score:.3f}")


    top_reranked_ids = [doc_id for doc_id, _ in reranked]
    assert 1 in top_reranked_ids or 2 in top_reranked_ids or 4 in top_reranked_ids

if __name__ == "__main__":
    test_reranker_trains_and_ranks()