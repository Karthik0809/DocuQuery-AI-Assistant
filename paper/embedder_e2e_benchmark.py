#!/usr/bin/env python3
"""
Supplementary benchmark: embedder comparison + end-to-end latency.
Run from repo root:  python paper/embedder_e2e_benchmark.py

1. Embedder comparison — dense-only Recall@5 on the paraphrase scenario for
   all-MiniLM-L6-v2 (22M params) vs all-mpnet-base-v2 (110M params), to test
   whether the weak dense results are an embedder-capacity issue.
2. End-to-end latency — query -> hybrid retrieval -> extractive QA answer
   (deepset/roberta-base-squad2), i.e. the full local answer path with no API.

No API keys required. Writes paper/e2e_results.txt
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sentence_transformers import SentenceTransformer
from rag_engine import AdvancedVectorStore, EnhancedQAModel
from micro_benchmark import build_corpus, rank_dense, NUM_CHUNKS, RNG_SEED


def dense_recall_at_5(embedder_name: str, chunks, qa):
    emb = SentenceTransformer(embedder_name)
    store = AdvancedVectorStore(emb, pinecone_api_key=None)
    t0 = time.perf_counter()
    store.build(chunks)
    build_s = time.perf_counter() - t0
    n = len(store.chunks)
    hits = []
    for q, gold in qa:
        ranked = rank_dense(store, q, n)
        hits.append(int(gold in ranked[:5]))
    return float(np.mean(hits)), build_s, store


def main():
    chunks, qa_id, qa_sem, qa_ocr, qa_ml = build_corpus(NUM_CHUNKS, RNG_SEED)
    lines = []

    # ── 1. Embedder comparison (dense-only, paraphrase scenario) ──────────────
    print("=== Embedder comparison: dense-only Recall@5, paraphrase queries ===")
    results = {}
    store_minilm = None
    for name in ["sentence-transformers/all-MiniLM-L6-v2",
                 "sentence-transformers/all-mpnet-base-v2"]:
        r5, build_s, store = dense_recall_at_5(name, chunks, qa_sem)
        results[name] = r5
        if "MiniLM" in name:
            store_minilm = store
        print(f"  {name.split('/')[-1]:25s} Recall@5={r5:.3f}  (index build {build_s:.1f}s)")
        lines.append(f"dense_recall5[{name.split('/')[-1]}] = {r5:.3f}")

    # ── 2. End-to-end latency: retrieval + extractive QA ─────────────────────
    print("\n=== End-to-end latency: query -> hybrid retrieval -> extractive QA ===")
    qa_model = EnhancedQAModel()
    e2e, retr_only = [], []
    for q, _ in qa_sem[:50]:
        t0 = time.perf_counter()
        hits, retr_t = store_minilm.search(q, top_k=5, use_expansion=True, rerank=True)
        retr_only.append(retr_t * 1000.0)
        qa_model.answer(q, hits)
        e2e.append((time.perf_counter() - t0) * 1000.0)

    e2e = np.array(e2e); retr_only = np.array(retr_only)
    print(f"  Retrieval only : mean={retr_only.mean():.1f}ms  median={np.median(retr_only):.1f}ms  p95={np.percentile(retr_only,95):.1f}ms")
    print(f"  End-to-end     : mean={e2e.mean():.1f}ms  median={np.median(e2e):.1f}ms  p95={np.percentile(e2e,95):.1f}ms")
    lines.append(f"retrieval_ms: mean={retr_only.mean():.1f} median={np.median(retr_only):.1f} p95={np.percentile(retr_only,95):.1f}")
    lines.append(f"e2e_ms:       mean={e2e.mean():.1f} median={np.median(e2e):.1f} p95={np.percentile(e2e,95):.1f}")

    out = os.path.join(os.path.dirname(__file__), "e2e_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
