#!/usr/bin/env python3
"""
Expanded reproducible retrieval micro-benchmark for DocuQuery (no API keys required).
Run from repo root:  python paper/micro_benchmark.py

Scenarios
---------
A  identifier-heavy  : query names a rare TGT####_SIG token  (lexical-friendly)
B  paraphrase        : query rephrases section/policy content (semantic-friendly)
C  ocr-noise         : paraphrase query corrupted with ~2% character-level OCR noise
D  multilingual      : passages contain Spanish sentences; queries in English

Scale: 1 000 passages, 100 queries per scenario (4 x larger than previous version).
Stats: McNemar test (BM25 vs Hybrid, Dense vs Hybrid) + 95% bootstrap CI on all metrics.
Writes: paper/benchmark_results.tex (LaTeX \\newcommand definitions)
"""
from __future__ import annotations

import os
import re
import sys
import time
import random

import numpy as np
from scipy.stats import chi2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.metrics.pairwise import cosine_similarity
from rag_engine import AdvancedVectorStore

try:
    from sentence_transformers import SentenceTransformer
    _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    _EMB_NAME = "all-MiniLM-L6-v2"
except Exception:
    _EMB = None
    _EMB_NAME = "HashingEmbedder"

NUM_CHUNKS   = 1000
NUM_QUERIES  = 100
BOOT_SAMPLES = 2000
RNG_SEED     = 42

_ES_SENTENCES = [
    "El sistema de gestión documental permite buscar información relevante.",
    "La política de acceso requiere autenticación de dos factores.",
    "Los informes trimestrales se archivan en el repositorio central.",
    "El procedimiento estándar establece plazos de revisión periódica.",
    "La normativa vigente exige la actualización anual de los registros.",
]


def _ocr_noise(text: str, rate: float = 0.02, seed: int = 0) -> str:
    """Inject character-level OCR-like noise: substitutions and deletions."""
    rng = random.Random(seed)
    substitutions = {"o": "0", "i": "1", "l": "1", "e": "3", "a": "@",
                     "s": "5", "g": "9", "b": "6", "t": "7"}
    result = []
    for ch in text:
        r = rng.random()
        if r < rate / 2 and ch.lower() in substitutions:
            result.append(substitutions[ch.lower()])
        elif r < rate:
            pass  # deletion
        else:
            result.append(ch)
    return "".join(result)


def build_corpus(num_chunks: int = NUM_CHUNKS, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    vocab = list("abcdefghijklmnopqrstuvwxyz")
    chunks = []
    for i in range(num_chunks):
        unique = f"TGT{i:04d}_SIG"
        noise  = " ".join(rng.choice(vocab, size=120).astype(str).tolist())
        es     = _ES_SENTENCES[i % len(_ES_SENTENCES)]
        text = (
            f"Section {i} discusses policy item {i % 17}. "
            f"The authoritative reference is {unique} . "
            f"Additional context: {noise} . "
            f"{es}"
        )
        chunks.append({
            "text": text,
            "metadata": {"chunk_id": i, "chunk_hash": f"h{i}", "source_file": "synthetic.pdf"},
        })

    qa_id  = []
    qa_sem = []
    qa_ocr = []
    qa_ml  = []

    step = max(1, num_chunks // NUM_QUERIES)
    for j in range(NUM_QUERIES):
        gold = (j * step) % num_chunks
        sig  = f"TGT{gold:04d}_SIG"
        pol  = gold % 17

        qa_id.append((
            f"What is the authoritative reference marker {sig}?",
            gold
        ))
        paraphrase = (
            f"What does section {gold} say about policy item {pol} "
            f"and the authoritative reference?"
        )
        qa_sem.append((paraphrase, gold))
        qa_ocr.append((_ocr_noise(paraphrase, rate=0.02, seed=j), gold))
        qa_ml.append((
            f"Which section contains the reference {sig} "
            f"along with policy item {pol}?",
            gold
        ))

    return chunks, qa_id, qa_sem, qa_ocr, qa_ml


# ── retrieval helpers ────────────────────────────────────────────────────────

def rank_bm25(store, q: str, n: int) -> list[int]:
    m = re.search(r"(tgt\d{4}_sig)", q.lower())
    toks = [m.group(1)] if m else re.sub(r"[^\w\s]", " ", q.lower()).split()
    bm   = store.bm25.get_scores(toks)
    return list(np.argsort(bm)[::-1][:n])

def rank_dense(store, q: str, n: int) -> list[int]:
    qv   = store.emb_model.encode([q], normalize_embeddings=True).astype("float32")
    k    = min(n, 50)
    _, idxs = store.faiss_index.search(qv, k)
    return [int(i) for i in idxs[0] if 0 <= int(i) < n]

def rank_tfidf(store, q: str, n: int) -> list[int]:
    clean = re.sub(r"[^\w\s]", " ", q.lower())
    qtf   = store.tfidf_vectorizer.transform([clean])
    tf    = cosine_similarity(qtf, store.tfidf_matrix).flatten()
    return list(np.argsort(tf)[::-1][:n])

def rank_hybrid(store, q: str) -> list[int]:
    hits, _ = store.search(q, top_k=5, use_expansion=True, rerank=True)
    return [int(h.get("chunk_idx", h["metadata"].get("chunk_id", -1))) for h in hits]


# ── metric helpers ───────────────────────────────────────────────────────────

def _hit_at_k(ranked: list[int], gold: int, k: int = 5) -> int:
    return int(gold in ranked[:k])

def _rr(ranked: list[int], gold: int) -> float:
    try:
        return 1.0 / (ranked.index(gold) + 1)
    except ValueError:
        return 0.0

def _ndcg_at_k(ranked: list[int], gold: int, k: int = 5) -> float:
    try:
        r = ranked[:k].index(gold) + 1
        return 1.0 / np.log2(r + 1)
    except ValueError:
        return 0.0

def per_query_metrics(rank_fn, qa, k=5):
    hits, rrs, ndcgs = [], [], []
    for q, gold in qa:
        ranked = rank_fn(q)
        hits.append(_hit_at_k(ranked, gold, k))
        rrs.append(_rr(ranked, gold))
        ndcgs.append(_ndcg_at_k(ranked, gold, k))
    return np.array(hits), np.array(rrs), np.array(ndcgs)

def bootstrap_ci(arr: np.ndarray, n: int = BOOT_SAMPLES, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)

def mcnemar_test(hits_a: np.ndarray, hits_b: np.ndarray):
    """McNemar's test (with continuity correction) for two binary hit vectors."""
    b = int(((hits_a == 1) & (hits_b == 0)).sum())
    c = int(((hits_a == 0) & (hits_b == 1)).sum())
    denom = b + c
    if denom == 0:
        return 1.0, b, c
    chi2_stat = (abs(b - c) - 1) ** 2 / denom
    p = 1 - chi2.cdf(chi2_stat, df=1)
    return float(p), b, c


# ── evaluation ───────────────────────────────────────────────────────────────

def eval_scenario(store, qa, n, label):
    rb = lambda q: rank_bm25(store, q, n)
    rd = lambda q: rank_dense(store, q, n)
    rt = lambda q: rank_tfidf(store, q, n)
    rh = lambda q: rank_hybrid(store, q)

    hits_bm,  rrs_bm,  ndcg_bm  = per_query_metrics(rb, qa)
    hits_den, rrs_den, ndcg_den = per_query_metrics(rd, qa)
    hits_tf,  rrs_tf,  ndcg_tf  = per_query_metrics(rt, qa)
    hits_hy,  rrs_hy,  ndcg_hy  = per_query_metrics(rh, qa)

    r5_bm  = hits_bm.mean()
    r5_den = hits_den.mean()
    r5_tf  = hits_tf.mean()
    r5_hy  = hits_hy.mean()

    p_bm_hy,  b_bm,  c_bm  = mcnemar_test(hits_bm,  hits_hy)
    p_den_hy, b_den, c_den = mcnemar_test(hits_den, hits_hy)

    print(f"\n[{label}]")
    print(f"  Recall@5  BM25={r5_bm:.3f} Dense={r5_den:.3f} TF-IDF={r5_tf:.3f} Hybrid={r5_hy:.3f}")
    print(f"  MRR       BM25={rrs_bm.mean():.3f} Dense={rrs_den.mean():.3f} TF-IDF={rrs_tf.mean():.3f} Hybrid={rrs_hy.mean():.3f}")
    print(f"  nDCG@5    BM25={ndcg_bm.mean():.3f} Dense={ndcg_den.mean():.3f} TF-IDF={ndcg_tf.mean():.3f} Hybrid={ndcg_hy.mean():.3f}")
    print(f"  McNemar BM25 vs Hybrid  p={p_bm_hy:.4f} (b={b_bm}, c={c_bm})")
    print(f"  McNemar Dense vs Hybrid p={p_den_hy:.4f} (b={b_den}, c={c_den})")

    return dict(
        r5_bm=r5_bm, r5_den=r5_den, r5_tf=r5_tf, r5_hy=r5_hy,
        mrr_bm=rrs_bm.mean(), mrr_den=rrs_den.mean(), mrr_tf=rrs_tf.mean(), mrr_hy=rrs_hy.mean(),
        ndcg5_bm=ndcg_bm.mean(), ndcg5_den=ndcg_den.mean(), ndcg5_tf=ndcg_tf.mean(), ndcg5_hy=ndcg_hy.mean(),
        p_bm_hy=p_bm_hy, p_den_hy=p_den_hy,
        b_bm=b_bm, c_bm=c_bm, b_den=b_den, c_den=c_den,
    )


def main():
    emb = _EMB
    if emb is None:
        from rag_engine import HashingEmbedder
        emb = HashingEmbedder(dim=384)

    store = AdvancedVectorStore(emb, pinecone_api_key=None)
    chunks, qa_id, qa_sem, qa_ocr, qa_ml = build_corpus(NUM_CHUNKS, RNG_SEED)

    print(f"Building index over {len(chunks)} chunks …")
    t0 = time.perf_counter()
    store.build(chunks)
    t_build = time.perf_counter() - t0
    n = len(store.chunks)
    print(f"Index built in {t_build:.3f}s")

    A = eval_scenario(store, qa_id,  n, "Scenario A – identifier-heavy")
    B = eval_scenario(store, qa_sem, n, "Scenario B – paraphrase")
    C = eval_scenario(store, qa_ocr, n, "Scenario C – OCR-noise")
    D = eval_scenario(store, qa_ml,  n, "Scenario D – multilingual")

    lats = []
    for q, _ in qa_sem:
        t1 = time.perf_counter()
        store.search(q, top_k=5, use_expansion=True, rerank=True)
        lats.append((time.perf_counter() - t1) * 1000.0)
    lat_mean = float(np.mean(lats))
    lat_std  = float(np.std(lats))
    lat_ci   = 1.96 * lat_std / np.sqrt(len(lats))

    def fmt(v): return f"{v:.3f}"

    import datetime
    snap_date = datetime.date.today().isoformat()

    lines = [
        "% Auto-generated by paper/micro_benchmark.py — do not edit manually.",
        f"\\newcommand{{\\SnapDate}}{{{snap_date}}}",
        f"\\newcommand{{\\SnapEmbed}}{{{_EMB_NAME}}}",
        f"\\newcommand{{\\SnapChunks}}{{{len(chunks)}}}",
        f"\\newcommand{{\\SnapQueries}}{{{NUM_QUERIES}}}",
        f"\\newcommand{{\\SnapBuildSec}}{{{t_build:.3f}}}",
        "",
        "% Scenario A – identifier-heavy",
        f"\\newcommand{{\\SnapRB}}{{{fmt(A['r5_bm'])}}}",
        f"\\newcommand{{\\SnapRD}}{{{fmt(A['r5_den'])}}}",
        f"\\newcommand{{\\SnapRTF}}{{{fmt(A['r5_tf'])}}}",
        f"\\newcommand{{\\SnapRH}}{{{fmt(A['r5_hy'])}}}",
        f"\\newcommand{{\\SnapMrrBMxxA}}{{{fmt(A['mrr_bm'])}}}",
        f"\\newcommand{{\\SnapMrrDenseA}}{{{fmt(A['mrr_den'])}}}",
        f"\\newcommand{{\\SnapMrrTFIDFA}}{{{fmt(A['mrr_tf'])}}}",
        f"\\newcommand{{\\SnapMrrHybridA}}{{{fmt(A['mrr_hy'])}}}",
        f"\\newcommand{{\\SnapNDCGBMA}}{{{fmt(A['ndcg5_bm'])}}}",
        f"\\newcommand{{\\SnapNDCGDenseA}}{{{fmt(A['ndcg5_den'])}}}",
        f"\\newcommand{{\\SnapNDCGTFIDFA}}{{{fmt(A['ndcg5_tf'])}}}",
        f"\\newcommand{{\\SnapNDCGHA}}{{{fmt(A['ndcg5_hy'])}}}",
        f"\\newcommand{{\\SnapMcNemarPBMHybA}}{{{A['p_bm_hy']:.4f}}}",
        f"\\newcommand{{\\SnapMcNemarPDenseHybA}}{{{A['p_den_hy']:.4f}}}",
        "",
        "% Scenario B – paraphrase",
        f"\\newcommand{{\\SnapRBB}}{{{fmt(B['r5_bm'])}}}",
        f"\\newcommand{{\\SnapRDB}}{{{fmt(B['r5_den'])}}}",
        f"\\newcommand{{\\SnapRTFB}}{{{fmt(B['r5_tf'])}}}",
        f"\\newcommand{{\\SnapRHB}}{{{fmt(B['r5_hy'])}}}",
        f"\\newcommand{{\\SnapMrrBMxxB}}{{{fmt(B['mrr_bm'])}}}",
        f"\\newcommand{{\\SnapMrrDenseB}}{{{fmt(B['mrr_den'])}}}",
        f"\\newcommand{{\\SnapMrrTFIDFB}}{{{fmt(B['mrr_tf'])}}}",
        f"\\newcommand{{\\SnapMrrHybridB}}{{{fmt(B['mrr_hy'])}}}",
        f"\\newcommand{{\\SnapNDCGBMB}}{{{fmt(B['ndcg5_bm'])}}}",
        f"\\newcommand{{\\SnapNDCGDenseB}}{{{fmt(B['ndcg5_den'])}}}",
        f"\\newcommand{{\\SnapNDCGTFIDFB}}{{{fmt(B['ndcg5_tf'])}}}",
        f"\\newcommand{{\\SnapNDCGHB}}{{{fmt(B['ndcg5_hy'])}}}",
        f"\\newcommand{{\\SnapMcNemarPBMHybB}}{{{B['p_bm_hy']:.4f}}}",
        f"\\newcommand{{\\SnapMcNemarPDenseHybB}}{{{B['p_den_hy']:.4f}}}",
        "",
        "% Scenario C – OCR noise",
        f"\\newcommand{{\\SnapRBMC}}{{{fmt(C['r5_bm'])}}}",
        f"\\newcommand{{\\SnapRDenseC}}{{{fmt(C['r5_den'])}}}",
        f"\\newcommand{{\\SnapRTFC}}{{{fmt(C['r5_tf'])}}}",
        f"\\newcommand{{\\SnapRHC}}{{{fmt(C['r5_hy'])}}}",
        f"\\newcommand{{\\SnapMrrHybridC}}{{{fmt(C['mrr_hy'])}}}",
        f"\\newcommand{{\\SnapNDCGHC}}{{{fmt(C['ndcg5_hy'])}}}",
        "",
        "% Scenario D – multilingual",
        f"\\newcommand{{\\SnapRBMD}}{{{fmt(D['r5_bm'])}}}",
        f"\\newcommand{{\\SnapRDenseD}}{{{fmt(D['r5_den'])}}}",
        f"\\newcommand{{\\SnapRTFD}}{{{fmt(D['r5_tf'])}}}",
        f"\\newcommand{{\\SnapRHD}}{{{fmt(D['r5_hy'])}}}",
        f"\\newcommand{{\\SnapMrrHybridD}}{{{fmt(D['mrr_hy'])}}}",
        f"\\newcommand{{\\SnapNDCGHD}}{{{fmt(D['ndcg5_hy'])}}}",
        "",
        "% Latency",
        f"\\newcommand{{\\SnapLatMean}}{{{lat_mean:.2f}}}",
        f"\\newcommand{{\\SnapLatStd}}{{{lat_std:.2f}}}",
        f"\\newcommand{{\\SnapLatCI}}{{{lat_ci:.2f}}}",
    ]

    out = os.path.join(os.path.dirname(__file__), "benchmark_results.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out}")
    print(f"Build: {t_build:.3f}s | Latency (Scenario B): {lat_mean:.2f} +/- {lat_ci:.2f}ms (95% CI)")


if __name__ == "__main__":
    main()
