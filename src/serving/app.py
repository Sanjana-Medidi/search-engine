"""
FastAPI application serving the two-stage search engine.
Endpoints:
  POST /search        - Search with query, returns top-k results
  GET  /health        - Health check
  POST /index         - Build the index from documents
  POST /train-ranker  - Train/re-train the re-ranker
"""

import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog

from src.indexer.indexer import InvertedIndex
from src.retrieval.retriever import BM25Retriever
from src.reranker.lightgbm_ranker import LightGBMRanker

logger = structlog.get_logger()

# Global state
index: Optional[InvertedIndex] = None
retriever: Optional[BM25Retriever] = None
ranker: Optional[LightGBMRanker] = None


class Document(BaseModel):
    doc_id: int
    text: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    doc_id: int
    score: float
    text: str
    rank: int


class SearchResponse(BaseModel):
    query: str
    num_results: int
    results: list[SearchResult]
    latency_ms: float


class IndexRequest(BaseModel):
    documents: list[Document]
    save_index_path: Optional[str] = None


class IndexResponse(BaseModel):
    num_docs: int
    vocab_size: int
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global index, retriever
    logger.info("Initializing search engine...")

    index_path = Path("data/index.json")
    if index_path.exists():
        index = InvertedIndex.load(index_path)
        retriever = BM25Retriever(index)
        logger.info("Index loaded from disk", num_docs=index.get_stats().num_docs)
    else:
        index = InvertedIndex()
        retriever = BM25Retriever(index)
        logger.info("Initialized empty index")
    
    yield
    logger.info("Shutting down search engine")


app = FastAPI(
    title="Search Engine API",
    description="Two-stage search with BM25 + LightGBM re-ranking",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """System health check."""
    if index is None:
        return {"status": "unhealthy", "message": "Index not initialized"}
    return {
        "status": "healthy",
        "num_docs": index.get_stats().num_docs,
        "vocab_size": len(index.index),
    }


@app.post("/index", response_model=IndexResponse)
async def build_index(request: IndexRequest):
    """Build the inverted index from a list of documents."""
    global index, retriever, ranker

    index = InvertedIndex()
    for doc in request.documents:
        index.add_document(doc.doc_id, doc.text)

    retriever = BM25Retriever(index)
    ranker = None

    if request.save_index_path:
        index.save(Path(request.save_index_path))

    stats = index.get_stats()
    return IndexResponse(
        num_docs=stats.num_docs,
        vocab_size=len(index.index),
        message=f"Indexed {stats.num_docs} documents",
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search using the two-stage retrieval and re-ranking pipeline."""
    if index is None or retriever is None:
        raise HTTPException(status_code=400, detail="Index not initialized.")

    start_time = time.time()
    
    # Stage 1: BM25
    candidates = retriever.retrieve(request.query, top_k=100)
    candidate_ids = [doc_id for doc_id, _ in candidates]

    # Stage 2: Re-ranker
    if ranker is not None and candidate_ids:
        reranked = ranker.rerank(request.query, candidate_ids, top_k=request.top_k)
    else:
        reranked = [(doc_id, score) for doc_id, score in candidates[:request.top_k]]

    # Response construction
    results = [
        SearchResult(
            doc_id=doc_id,
            score=float(score),
            text=index.get_document(doc_id),
            rank=i + 1,
        )
        for i, (doc_id, score) in enumerate(reranked)
    ]

    return SearchResponse(
        query=request.query,
        num_results=len(results),
        results=results,
        latency_ms=(time.time() - start_time) * 1000,
    )


@app.post("/train-ranker")
async def train_ranker(training_data: list[dict]):
    """Train the re-ranker on labeled query-doc triples."""
    global index, retriever, ranker

    if index is None or retriever is None:
        raise HTTPException(status_code=400, detail="Index not initialized.")

    triples = [(item["query"], item["doc_id"], item["relevance"]) for item in training_data]
    ranker = LightGBMRanker(index, retriever)
    ranker.train(triples)

    return {"message": f"Trained ranker on {len(triples)} examples"}


@app.get("/")
async def root():
    return {"status": "online", "service": "Search Engine API"}