"""
Build an evaluation corpus that GUARANTEES gold (relevant) passages are
present, and splits queries into non-overlapping train/test sets BEFORE
any triples or metrics are computed.

This fixes two problems with random passage subsampling:
1. Only ~26/100k queries had their gold passage survive a random 10k sample
   (10k / 8.8M passages ≈ 0.11% survival rate per gold passage).
2. Training the reranker and evaluating it on the SAME query set causes
   leakage — the model "learns" labels it's then graded on.

Usage:
    python -m src.data.build_eval_corpus \
        --collection data/msmarco/collection.tsv \
        --queries data/msmarco/queries.dev.small.tsv \
        --qrels data/msmarco/qrels.dev.small.tsv \
        --num-queries 1000 \
        --num-distractors 9000 \
        --train-frac 0.7 \
        --out-dir data/eval_v2

Output:
    data/eval_v2/passages.tsv        <- gold passages + random distractors
    data/eval_v2/queries.train.tsv   <- queries to build reranker triples from
    data/eval_v2/qrels.train.tsv
    data/eval_v2/queries.test.tsv    <- HELD OUT, never seen during training
    data/eval_v2/qrels.test.tsv
"""
import argparse
import random
from pathlib import Path
from collections import defaultdict


def load_qrels(path):
    """qid -> set of relevant pids. Handles 3-col (qid,pid,rel) and
    4-col (qid,0,pid,rel) TREC-style qrels files."""
    qrels = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                qid, _, pid, rel = parts
            elif len(parts) == 3:
                qid, pid, rel = parts
            else:
                continue
            if int(rel) > 0:
                qrels[qid].add(pid)
    return qrels


def load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[qid] = text
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--num-queries", type=int, default=1000,
                     help="How many judged queries to guarantee in the corpus")
    ap.add_argument("--num-distractors", type=int, default=9000,
                     help="Random extra (non-relevant) passages to pad the corpus with")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out-dir", default="data/eval_v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading qrels...")
    qrels = load_qrels(args.qrels)
    print(f"  {len(qrels)} queries have judgments")

    print("Loading queries...")
    queries = load_queries(args.queries)

    usable_qids = [qid for qid in qrels if qid in queries]
    print(f"  {len(usable_qids)} queries have both judgments and text")
    if len(usable_qids) < args.num_queries:
        print(f"  NOTE: only {len(usable_qids)} usable queries available, "
              f"using all of them instead of {args.num_queries}")

    random.shuffle(usable_qids)
    chosen_qids = usable_qids[: args.num_queries]
    print(f"  Selected {len(chosen_qids)} queries to guarantee in corpus")

    # Split BEFORE building anything else -> train/test queries never overlap
    split_idx = int(len(chosen_qids) * args.train_frac)
    train_qids = chosen_qids[:split_idx]
    test_qids = chosen_qids[split_idx:]
    print(f"  Train queries: {len(train_qids)} | Test queries: {len(test_qids)}")

    gold_pids = set()
    for qid in chosen_qids:
        gold_pids.update(qrels[qid])
    print(f"  {len(gold_pids)} gold passages must be included in corpus")

    print("Scanning collection.tsv (single pass) for gold passages + sampling distractors...")
    gold_lines = {}
    distractor_pool = []
    k = args.num_distractors
    distractor_seen = 0  # only counts non-gold lines, for correct reservoir sampling

    with open(args.collection, encoding="utf-8") as f:
        for line in f:
            pid = line.split("\t", 1)[0]
            if pid in gold_pids:
                gold_lines[pid] = line
            else:
                if len(distractor_pool) < k:
                    distractor_pool.append(line)
                else:
                    j = random.randint(0, distractor_seen)
                    if j < k:
                        distractor_pool[j] = line
                distractor_seen += 1

    print(f"  Found {len(gold_lines)}/{len(gold_pids)} gold passages in collection")
    missing = gold_pids - set(gold_lines)
    if missing:
        print(f"  WARNING: {len(missing)} gold pids not found in collection.tsv "
              f"(those queries will have incomplete judgments)")

    corpus_path = out_dir / "passages.tsv"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in gold_lines.values():
            f.write(line if line.endswith("\n") else line + "\n")
        for line in distractor_pool:
            f.write(line if line.endswith("\n") else line + "\n")
    total = len(gold_lines) + len(distractor_pool)
    print(f"  Wrote {total} passages -> {corpus_path}")

    def write_split(name, qids):
        with open(out_dir / f"queries.{name}.tsv", "w", encoding="utf-8") as fq, \
             open(out_dir / f"qrels.{name}.tsv", "w", encoding="utf-8") as fr:
            for qid in qids:
                fq.write(f"{qid}\t{queries[qid]}\n")
                for pid in qrels[qid]:
                    fr.write(f"{qid}\t0\t{pid}\t1\n")

    write_split("train", train_qids)
    write_split("test", test_qids)
    print(f"  Wrote queries.train/test.tsv + qrels.train/test.tsv -> {out_dir}")

    print("\nDone.")
    print("  -> Build reranker triples ONLY from queries.train.tsv / qrels.train.tsv")
    print("  -> Report final metrics ONLY on queries.test.tsv / qrels.test.tsv")
    print("  These two query sets are disjoint, so there is no leakage.")


if __name__ == "__main__":
    main()