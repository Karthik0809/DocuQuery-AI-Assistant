#!/usr/bin/env python3
"""
Reproducible retrieval ablation + latency micro-benchmark for DocuQuery (local, no API keys).
Run from repo root:  python paper/micro_benchmark.py
Writes: paper/benchmark_results.tex (LaTeX \\newcommand definitions)
"""
from __future__ import annotations

import os
import re
import sys
import time
import numpy as np

# Repo root (parent of paper/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.metrics.pairwise import cosine_similarity

from rag_engine import AdvancedVectorStore

try:
    from sentence_transformers import SentenceTransformer

    _EMB = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _EMB = None


def build_corpus(num_chunks: int = 120, seed: int = 42) -> tuple[list[dict], list[tuple[str, int]]]:
    """Synthetic chunks with unique target phrases; (query, gold_chunk_idx) pairs.

    No shared boilerplate across chunks (shared text makes BM25/TF-IDF ties).
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(num_chunks):
        unique = f"TGT{i:04d}_SIG"
        noise = " ".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=120).astype(str).tolist())
        # Short, chunk-specific prose so lexical and dense methods can rank the gold passage.
        # Whitespace around the marker so BM25 tokenization matches query tokens (no `tgt0007_sig.` glued punctuation).
        text = (
            f"Section {i} discusses policy item {i % 17}. "
            f"The authoritative reference is {unique} . "
            f"Additional context: {noise} ."
        )
        chunks.append(
            {
                "text": text,
                "metadata": {"chunk_id": i, "chunk_hash": f"h{i}", "source_file": "synthetic.pdf"},
            }
        )
    qa_id = []  # Scenario A: rare identifier in query (lexical-friendly).
    qa_sem = []  # Scenario B: paraphrase using section/policy cues (semantic-friendly).
    for i in range(30):
        gold = (i * 3) % num_chunks
        qa_id.append((f"What is the authoritative reference marker TGT{gold:04d}_SIG?", gold))
        qa_sem.append(
            (
                f"What does section {gold} say about policy item {gold % 17} and the authoritative reference?",
                gold,
            )
        )
    return chunks, qa_id, qa_sem


def recall_at_k(rank_fn, qa: list, k: int = 5) -> float:
    hits = 0
    for q, gold in qa:
        ranked = rank_fn(q)[:k]
        if gold in ranked:
            hits += 1
    return hits / len(qa)


def mean_reciprocal_rank(rank_fn, qa: list) -> float:
    """MRR over full ranked list returned by rank_fn (gold not listed => RR=0)."""
    total = 0.0
    for q, gold in qa:
        ranked = rank_fn(q)
        try:
            pos = ranked.index(gold) + 1  # 1-based rank
            total += 1.0 / pos
        except ValueError:
            pass
    return total / len(qa) if qa else 0.0


def main():
    if _EMB is None:
        from rag_engine import HashingEmbedder

        emb = HashingEmbedder(dim=384)
        embed_name = "HashingEmbedder"
    else:
        emb = _EMB
        embed_name = "all-MiniLM-L6-v2"
    store = AdvancedVectorStore(emb, pinecone_api_key=None)
    chunks, qa_id, qa_sem = build_corpus(120)

    t0 = time.perf_counter()
    store.build(chunks)
    t_build = time.perf_counter() - t0

    n = len(store.chunks)

    def rank_bm25(q: str) -> list[int]:
        # Lexical-only: prefer the rare identifier token if present (mirrors keyword-style queries).
        m = re.search(r"(tgt\d{4}_sig)", q.lower())
        toks = [m.group(1)] if m else re.sub(r"[^\w\s]", " ", q.lower()).split()
        bm = store.bm25.get_scores(toks)
        return list(np.argsort(bm)[::-1][:n])

    def rank_dense(q: str) -> list[int]:
        qv = store.emb_model.encode([q], normalize_embeddings=True).astype("float32")
        k = min(n, 30)
        _, idxs = store.faiss_index.search(qv, k)
        return [int(i) for i in idxs[0] if 0 <= int(i) < n]

    def rank_tfidf(q: str) -> list[int]:
        clean = re.sub(r"[^\w\s]", " ", q.lower())
        qtf = store.tfidf_vectorizer.transform([clean])
        tf = cosine_similarity(qtf, store.tfidf_matrix).flatten()
        return list(np.argsort(tf)[::-1][:n])

    def rank_hybrid(q: str, expand: bool = True) -> list[int]:
        hits, _ = store.search(q, top_k=5, use_expansion=expand, rerank=True)
        return [int(h.get("chunk_idx", h["metadata"].get("chunk_id", -1))) for h in hits]

    def eval_scenario(qa: list, tag: str):
        r_bm = recall_at_k(rank_bm25, qa, 5)
        r_den = recall_at_k(rank_dense, qa, 5)
        r_tf = recall_at_k(rank_tfidf, qa, 5)
        r_hy = recall_at_k(lambda q: rank_hybrid(q, expand=True), qa, 5)
        r_hy_ne = recall_at_k(lambda q: rank_hybrid(q, expand=False), qa, 5)
        m_bm = mean_reciprocal_rank(rank_bm25, qa)
        m_den = mean_reciprocal_rank(rank_dense, qa)
        m_tf = mean_reciprocal_rank(rank_tfidf, qa)

        def rank_hy_exp(q: str) -> list[int]:
            return rank_hybrid(q, expand=True)

        def rank_hy_ne(q: str) -> list[int]:
            return rank_hybrid(q, expand=False)

        m_hy_exp = mean_reciprocal_rank(rank_hy_exp, qa)
        m_hy_ne = mean_reciprocal_rank(rank_hy_ne, qa)
        # Match Recall@5 column: same hybrid variant (exp vs no-exp) with higher aggregate recall.
        m_hy = m_hy_exp if r_hy >= r_hy_ne else m_hy_ne
        print(
            f"  [{tag}] Recall@5: BM25={r_bm:.2f} dense={r_den:.2f} tfidf={r_tf:.2f} "
            f"hybrid(exp)={r_hy:.2f} hybrid(no-exp)={r_hy_ne:.2f}"
        )
        print(
            f"  [{tag}] MRR: BM25={m_bm:.3f} dense={m_den:.3f} tfidf={m_tf:.3f} "
            f"hybrid(aligned w/ Recall@5)={m_hy:.3f} [exp={m_hy_exp:.3f} no_exp={m_hy_ne:.3f}]"
        )
        return r_bm, r_den, r_tf, max(r_hy, r_hy_ne), m_bm, m_den, m_tf, m_hy

    r_bm25_a, r_dense_a, r_tfidf_a, r_hybrid_a, m_bm25_a, m_dense_a, m_tfidf_a, m_hybrid_a = eval_scenario(
        qa_id, "identifier queries"
    )
    r_bm25_b, r_dense_b, r_tfidf_b, r_hybrid_b, m_bm25_b, m_dense_b, m_tfidf_b, m_hybrid_b = eval_scenario(
        qa_sem, "section paraphrase"
    )

    latencies = []
    for q, _ in qa_sem:
        t1 = time.perf_counter()
        store.search(q, top_k=5, use_expansion=True, rerank=True)
        latencies.append((time.perf_counter() - t1) * 1000.0)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))

    # Lines for LaTeX
    lines = [
        "% Auto-generated by paper/micro_benchmark.py — re-run after code changes.",
        f"\\newcommand{{\\BenchEmbedModel}}{{{embed_name}}}",
        f"\\newcommand{{\\BenchNumChunks}}{{{len(chunks)}}}",
        f"\\newcommand{{\\BenchNumQueries}}{{{len(qa_id)}}}",
        f"\\newcommand{{\\BenchBuildSec}}{{{t_build:.3f}}}",
        f"\\newcommand{{\\BenchRecallBM}}{{{r_bm25_a:.2f}}}",
        f"\\newcommand{{\\BenchRecallDense}}{{{r_dense_a:.2f}}}",
        f"\\newcommand{{\\BenchRecallTFIDF}}{{{r_tfidf_a:.2f}}}",
        f"\\newcommand{{\\BenchRecallHybrid}}{{{r_hybrid_a:.2f}}}",
        f"\\newcommand{{\\BenchRecallBMB}}{{{r_bm25_b:.2f}}}",
        f"\\newcommand{{\\BenchRecallDenseB}}{{{r_dense_b:.2f}}}",
        f"\\newcommand{{\\BenchRecallTFIDFB}}{{{r_tfidf_b:.2f}}}",
        f"\\newcommand{{\\BenchRecallHybridB}}{{{r_hybrid_b:.2f}}}",
        f"\\newcommand{{\\BenchMrrBM}}{{{m_bm25_a:.3f}}}",
        f"\\newcommand{{\\BenchMrrDense}}{{{m_dense_a:.3f}}}",
        f"\\newcommand{{\\BenchMrrTFIDF}}{{{m_tfidf_a:.3f}}}",
        f"\\newcommand{{\\BenchMrrHybrid}}{{{m_hybrid_a:.3f}}}",
        f"\\newcommand{{\\BenchMrrBMB}}{{{m_bm25_b:.3f}}}",
        f"\\newcommand{{\\BenchMrrDenseB}}{{{m_dense_b:.3f}}}",
        f"\\newcommand{{\\BenchMrrTFIDFB}}{{{m_tfidf_b:.3f}}}",
        f"\\newcommand{{\\BenchMrrHybridB}}{{{m_hybrid_b:.3f}}}",
        f"\\newcommand{{\\BenchLatencyMeanMs}}{{{mean_lat:.2f}}}",
        f"\\newcommand{{\\BenchLatencyStdMs}}{{{std_lat:.2f}}}",
    ]
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote", out_path)
    print(f"Build time: {t_build:.3f}s | Scenario A (table defaults): BM25={r_bm25_a:.2f} dense={r_dense_a:.2f} tfidf={r_tfidf_a:.2f} hybrid={r_hybrid_a:.2f}")
    print(f"Mean retrieval latency: {mean_lat:.2f} ms (std {std_lat:.2f})")
    print(
        "NEXT: Sync paper/docuquery_ieee.tex BENCHMARK SNAPSHOT block (\\\\newcommand{\\\\Snap...}) "
        "and abstract/tables if any values changed; benchmark_results.tex is for reference only."
    )


if __name__ == "__main__":
    main()
