from collections import defaultdict 
import re

class InvertedIndex:
    #{ "word": [doc_id1, doc_id2, ...] }
    def __init__(self) -> None:
        self.index: dict[str, list[int]] = defaultdict(list)
        self.documents: dict[int,str] = {}

    def tokenize(self,text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())
    
    def add_document(self, doc_id: int, text: str) -> None:
        self.documents[doc_id] = text
        tokens = self.tokenize(text)

        for token in set(tokens):
            self.index[token].append(doc_id)

    def search(self, query: str) -> list[int]:
        tokens= self.tokenize(query)
        if not tokens:
            return []
        
        return self.index.get(tokens[0],[])
    
