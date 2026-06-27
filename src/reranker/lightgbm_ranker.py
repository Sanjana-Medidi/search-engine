import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from src.indexer.indexer import InvertedIndex
from src.retrieval.retriever import BM25Retriever

class RankerFeatures:

    def __init__(self, index: InvertedIndex, retriever: BM25Retriever):
        self.index = index
        self.retriever = retriever

    def extract(self, query: str, doc_id: int) -> dict[str, float]:
        query_tokens = self.index.tokenize(query)
        doc_text = self.index.get_document(doc_id)
        doc_tokens = self.index.tokenize(doc_text)
        stats = self.index.get_stats()

        bm25_score = self.retriever.score(query, doc_id)
        query_length = len(query_tokens)
        doc_length = stats.doc_lengths.get(doc_id,0)

        doc_term_set = set(doc_tokens)
        term_overlap = sum(1 for t in query_tokens if t in doc_term_set)

        term_freqs = []
        for token in query_tokens:
            for posting in self.index.get_postings(token):
                if posting.doc_id == doc_id:
                    term_freqs.append(posting.term_freq)
        avg_tf = np.mean(term_freqs) if term_freqs else 0.0

        return {
            "bm25_score": bm25_score,
            "query_length": query_length,
            "doc_length": doc_length,
            "term_overlap": term_overlap,
            "avg_term_freq": avg_tf,
        }


class LightGBMRanker:
    def __init__(self, index: InvertedIndex, retriever: BM25Retriever):
        self.index = index
        self.retriever = retriever
        self.feature_extractor = RankerFeatures(index,retriever)
        self.model: lgb.Booster | None = None
        self.feature_names = [
            "bm25_score",
            "query_length",
            "doc_length",
            "term_overlap",
            "avg_term_freq",
        ]

    def _prepare_training_data(
            self, triples: list[tuple[str,int,int]] #(query, doc_id, relevance)
    ) -> tuple[np.ndarray, np.ndarray]:
        X=[]
        y=[]
        for query,doc_id, relevance in triples:
            features = self.feature_extractor.extract(query,doc_id)
            X.append([features[name] for name in self.feature_names])
            y.append(relevance)

        return np.array(X), np.array(y)
    
    def train(self, triples: list[tuple[str,int,int]]) -> None:
        X,y = self._prepare_training_data(triples)

        train_data = lgb.Dataset(X, label=y)

        params = {
            "objective": "binary",
            "metric": "binary_logloss", #lower logloss better model
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
        }
        self.model = lgb.train(params, train_data, num_boost_round=100)
    def rerank(self, query: str, candidates: list[int], top_k: 10) -> list[tuple[int,float]]: #(doc_id, reranked_score)
        """
        query: the search query
        candidates: list of doc_ids from Stage 1 (BM25)
        top_k: return top_k results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X =[]
        for doc_id in candidates:
            features = self.feature_extractor.extract(query, doc_id)
            X.append([features[name] for name in self.feature_names])
        X = np.array(X)

        scores = self.model.predict(X)

        ranked = sorted(zip(candidates, scores), key = lambda x: x[1], reverse=True)
        return ranked[:top_k]
        
    def save(self,path: Path) -> None:
        #saving model to disk
        if self.model is None:
                raise ValueError("No model to save. Train first.")
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path, index : InvertedIndex, retriever: BM25Retriever) -> "LightGBMRanker":
        ranker = cls(index, retriever)
        ranker.model = lgb.Booster(model_file=str(path))
        return ranker



