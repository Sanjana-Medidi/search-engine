import math
from src.indexer.indexer import InvertedIndex, Posting

class BM25Retriever:
    def __init__(self, index: InvertedIndex,k1: float =1.5, b: float =0.75) -> None:
        self.index = index
        self.k1 = k1
        self.b = b

    #Inverse Document Frequency
    def _idf(self, term: str) -> float:
        stats = self.index.get_stats()
        N = stats.num_docs
        doc_freq = len(self.index.get_postings(term))
        if doc_freq==0:
            return 0.0

        return math.log((N - doc_freq + 0.5)/(doc_freq + 0.5) + 1)
        
    def score(self, query: str, doc_id: int) ->float:
        stats = self.index.get_stats()
        dl = stats.doc_lengths.get(doc_id, 0) #document length
        avgdl = stats.avg_doc_length
        tokens = self.index.tokenize(query)
        total =0.0

        for token in tokens:
            idf = self._idf(token)
            if idf == 0.0:
                continue

            tf =0
            for posting in self.index.get_postings(token):
                if posting.doc_id == doc_id:
                    tf = posting.term_freq
                    break

            numerator = tf * (self.k1 +1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl/ avgdl if avgdl > 0 else 1) 
            total += idf * (numerator / denominator)
        return total

    def retrieve(self, query: str, top_k: int=100) -> list[tuple[int, float]]:

        tokens = self.index.tokenize(query)
        if not tokens:
            return []
        candidate_ids: set[int] = set()
        for token in tokens:
            for posting in self.index.get_postings(token):
                candidate_ids.add(posting.doc_id)

        scored = [(doc_id, self.score(query,doc_id)) for doc_id in candidate_ids]

        scored.sort(key= lambda x: x[1], reverse=True)
        return scored[:top_k]
    