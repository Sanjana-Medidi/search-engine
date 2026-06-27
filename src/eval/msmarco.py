from __future__ import annotations
import csv
from pathlib import Path
from datasets import load_dataset

_CACHE = Path("data/msmarco")

def _ensure_data() -> None:
    """Downloads MS MARCO from Hugging Face if not present locally."""
    if _CACHE.exists():
        return

    _CACHE.mkdir(parents=True, exist_ok=True)
    print("Downloading MS MARCO corpus and queries...")
    
    dataset = load_dataset("beir/msmarco", "corpus", split="corpus")
    queries = load_dataset("beir/msmarco", "queries", split="queries")
    
    with open(_CACHE / "collection.tsv", "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(f"{row['_id']}\t{row['text']}\n")
            
    with open(_CACHE / "queries.dev.small.tsv", "w", encoding="utf-8") as f:
        for row in queries:
            f.write(f"{row['_id']}\t{row['text']}\n")
    print("Data preparation complete.")

def iter_passages(max_passages: int | None = 100_000) -> iter[tuple[int, str]]:
    """Iterates over the collection file."""
    _ensure_data()
    path = _CACHE / "collection.tsv"
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_passages and i >= max_passages:
                break
            yield int(row[0]), row[1]

def load_queries() -> dict[int, str]:
    """Loads MS MARCO development queries."""
    path = _CACHE / "queries.dev.small.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found at {path}")
    
    queries: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries[int(row[0])] = row[1]
    return queries

def load_qrels() -> dict[int, dict[int, int]]:
    """Loads query-relevance judgments."""
    path = _CACHE / "qrels.dev.small.tsv"
    if not path.exists():
        raise FileNotFoundError(f"qrels.dev.small.tsv not found at {path}")
        
    qrels: dict[int, dict[int, int]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, pid, rel = int(row[0]), int(row[2]), int(row[3])
            qrels.setdefault(qid, {})[pid] = rel
    return qrels

def load_msmarco(
    max_passages: int | None = 100_000,
) -> tuple[dict[int, str], dict[int, str], dict[int, dict[int, int]]]:
    """
    Wrapper to load the MS MARCO subset.
    
    Returns:
        passages: {pid: text}
        queries: {qid: text}
        qrels: {qid: {pid: relevance}}
    """
    passages = dict(iter_passages(max_passages))
    queries = load_queries()
    qrels = load_qrels()

    # Filter qrels to include only queries with at least one passage in the loaded subset
    qrels = {
        qid: pmap for qid, pmap in qrels.items()
        if any(pid in passages for pid in pmap)
    }

    return passages, queries, qrels