from collections import defaultdict, Counter
from dataclasses import dataclass, field
import json
import re
from pathlib import Path

@dataclass 
class Posting:
    doc_id: int
    term_freq: int

@dataclass
class IndexStats:
    num_docs: int =0
    avg_doc_length: float = 0.0
    doc_lengths: dict[int,int] = field(default_factory=dict)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, list[Posting]] = defaultdict(list)
        self.documents: dict[int,str] = {}
        self._stats = IndexStats()

    def tokenize(self,text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())
    
    def add_document(self, doc_id: int, text: str) -> None:
        self.documents[doc_id] = text
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        for term, freq in tf.items():
            self.index[term].append(Posting(doc_id=doc_id, term_freq=freq))

        doc_len = len(tokens)
        self._stats.doc_lengths[doc_id] = doc_len
        self._stats.num_docs +=1
        n= self._stats.num_docs
        self._stats.avg_doc_length += (doc_len - self._stats.avg_doc_length)/n

    def get_postings(self, term: str) -> list[Posting]:
        return self.index.get(term, [])

    def get_document(self, doc_id: int) -> str:
        return self.documents.get(doc_id, "")

    def get_stats(self) -> IndexStats:
        return self._stats
    
    def search(self, query: str) -> list[int]:
        tokens= self.tokenize(query)
        if not tokens:
            return []
        
        postings_lists = []
        for token in tokens:
            postings = self.index.get(token)
            if postings is None:
                return []
            
            postings_lists.append(postings)
    
        postings_lists.sort(key=len)

        result_ids = {p.doc_id for p in postings_lists[0]}
        for postings in postings_lists[1:]:
            result_ids &= {p.doc_id for p in postings}
            if not result_ids:
                return []
            
        return sorted(result_ids)
    
    def save(self, path: Path) ->None:
        payload = {
            "index": {
                term: [{"doc_id": p.doc_id, "tf": p.term_freq} for p in postings]
                for term, postings in self.index.items()
            },
            "documents": {str(k): v for k,v in self.documents.items()},
            "stats": {
                "num_docs": self._stats.num_docs,
                "avg_doc_length": self._stats.avg_doc_length,
                "doc_lengths": {str(k): v for k,v in self._stats.doc_lengths.items()},
            },
        }
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(json.dumps(payload, indent=2))


    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        payload = json.loads(path.read_text())
        idx = cls()
        for term, postings_data in payload["index"].items():
            idx.index[term] = [
            Posting(doc_id=p["doc_id"], term_freq=p["tf"])
            for p in postings_data
            ]
        idx.documents = {int(k): v for k, v in payload["documents"].items()}
        stats = payload["stats"]
        idx._stats = IndexStats(
            num_docs=stats["num_docs"],
            avg_doc_length=stats["avg_doc_length"],
            doc_lengths={int(k): v for k, v in stats["doc_lengths"].items()},
        )
        return idx