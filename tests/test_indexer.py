from src.indexer.indexer import InvertedIndex

def test_index_basic():
    idx = InvertedIndex()
    idx.add_document(1, "The quick brown fox")
    idx.add_document(2,"The lazy dog" )

    results = idx.search("fox")
    assert 1 in results
    print("Test passed: Indexer works!")

if __name__ == "__main__":
    test_index_basic()