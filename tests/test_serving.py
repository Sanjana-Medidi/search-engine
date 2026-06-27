import pytest
from fastapi.testclient import TestClient
from src.serving.app import app
from src.indexer.indexer import InvertedIndex
from src.retrieval.retriever import BM25Retriever


@pytest.fixture
def client():
    """FastAPI test client with proper initialization."""
    # Manually initialize the app state (since TestClient doesn't run lifespan)
    import src.serving.app as app_module
    app_module.index = InvertedIndex()
    app_module.retriever = BM25Retriever(app_module.index)
    app_module.ranker = None
    
    return TestClient(app)


def test_health_before_indexing(client):
    """Health check when no index has been built yet."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Before calling /index endpoint, health is unhealthy
    assert data["status"] == "unhealthy"
    assert "message" in data


def test_search_empty_index(client):
    """Search on empty index returns 400 error."""
    response = client.post("/search", json={"query": "test"})
    # App correctly rejects search on empty index
    assert response.status_code == 400


def test_train_ranker(client):
    """Train the LightGBM ranker."""
    # First build index
    documents = [
        {"doc_id": 1, "text": "The quick brown fox jumps over the lazy dog"},
        {"doc_id": 2, "text": "A fast fox leaps across the field"},
        {"doc_id": 3, "text": "The dog was very lazy and slept all day"},
    ]
    client.post("/index", json={"documents": documents})

    # Train ranker with synthetic labels
    training_data = [
        {"query": "quick fox", "doc_id": 1, "relevance": 1},
        {"query": "quick fox", "doc_id": 2, "relevance": 1},
        {"query": "quick fox", "doc_id": 3, "relevance": 0},
    ]
    
    response = client.post("/train-ranker", json=training_data)
    assert response.status_code == 200
    data = response.json()
    assert "Trained ranker" in data["message"]
    print(f"✓ {data['message']}")


def test_full_pipeline(client):
    """Full pipeline: index -> train -> search."""
    # 1. Index
    documents = [
        {"doc_id": 1, "text": "The quick brown fox jumps over the lazy dog"},
        {"doc_id": 2, "text": "A fast fox leaps across the field"},
        {"doc_id": 3, "text": "The dog was very lazy and slept all day"},
        {"doc_id": 4, "text": "Foxes are quick animals that jump high"},
    ]
    client.post("/index", json={"documents": documents})
    print("✓ Index built")

    # 2. Train ranker
    training_data = [
        {"query": "quick fox", "doc_id": 1, "relevance": 1},
        {"query": "quick fox", "doc_id": 2, "relevance": 1},
        {"query": "quick fox", "doc_id": 3, "relevance": 0},
        {"query": "quick fox", "doc_id": 4, "relevance": 1},
    ]
    client.post("/train-ranker", json=training_data)
    print("✓ Ranker trained")

    # 3. Search
    response = client.post("/search", json={"query": "quick fox", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Search returned {data['num_results']} results in {data['latency_ms']:.1f}ms")
    
    for r in data["results"]:
        print(f"  #{r['rank']}: Doc {r['doc_id']} (score {r['score']:.3f}) - {r['text'][:50]}")
