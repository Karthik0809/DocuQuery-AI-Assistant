# RAG Engine module for Enhanced RAG Chatbot
import re
import hashlib
import time
import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from config import TARGET_TOKENS, OVERLAP_SENTENCES, BOUNDARY_DROP, TOK_CHARS_RATIO
from utils import safe_sent_tokenize, MortgageHeuristics

class HierarchicalChunker:
    def __init__(self, embed_model, target_tokens=TARGET_TOKENS, overlap_sentences=OVERLAP_SENTENCES, boundary_drop=BOUNDARY_DROP, tok_chars_ratio=TOK_CHARS_RATIO):
        self.emb = embed_model
        self.target_tokens = target_tokens
        self.overlap = overlap_sentences
        self.boundary_drop = boundary_drop
        self.tok_chars_ratio = tok_chars_ratio
    
    def _est_tokens(self, s): 
        return max(1, int(len(s)*self.tok_chars_ratio))
    
    def split(self, raw_text, meta):
        text = re.sub(r'--- Page \d+.*?---\n?', '', raw_text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.(\s+)([A-Z])', r'.\n\2', text)
        chunks=[]
        for size,label in [(500,"large"),(300,"medium"),(150,"small")]:
            chunks.extend(self._mk_chunks(text, meta, size, label))
        return chunks
    
    def _mk_chunks(self, text, meta, target_tokens, chunk_type):
        sents = safe_sent_tokenize(text) or [text]
        if len(sents) < 2:
            return [{"text": text, "metadata": {"chunk_id": 0, "chunk_type": chunk_type,
                                                  "source_file": meta.get("file"), "source_path": meta.get("path"),
                                                  "chunk_hash": hashlib.md5(text.encode()).hexdigest()[:10],}}]
        sent_vecs = self.emb.encode(sents, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
        chunks=[]; cur=[]; curv=[]; prev=None
        def flush():
            if not cur: return
            t=" ".join(cur).strip(); chunks.append({"text": t, "metadata": {
                "chunk_id": len(chunks), "chunk_type": chunk_type,
                "source_file": meta.get("file"), "source_path": meta.get("path"),
                "chunk_hash": hashlib.md5(t.encode()).hexdigest()[:10],
                "token_count": self._est_tokens(t), "sentence_count": len(cur)
            }})
        for s,v in zip(sents, sent_vecs):
            if not cur: cur=[s]; curv=[v]; prev=1.0; continue
            sim=float(np.dot(v, curv[-1])); budget=self._est_tokens(" ".join(cur+[s]))
            boundary = (prev - sim) > self.boundary_drop or budget > target_tokens
            if boundary:
                flush(); ov = cur[-self.overlap:] if len(cur)>=self.overlap else cur
                cur = ov + [s]
                curv = ([self.emb.encode([o], normalize_embeddings=True)[0] for o in ov] if ov else []) + [v]
            else:
                cur.append(s); curv.append(v)
            prev=sim
        flush(); return chunks

class QueryExpander:
    def __init__(self):
        try: 
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except: 
            self.stop_words = {"the","a","an","and","or","but","in","on","at","to","for","of","with","by"}
    
    def expand(self, q):
        words = [w for w in q.lower().split() if w not in self.stop_words]
        aug=[]
        for w in words:
            if w.endswith('s') and len(w)>4: aug.append(w[:-1])
            elif not w.endswith('s'): aug.append(w+'s')
            if w.endswith('ing'): aug.append(w[:-3])
            if w.endswith('ed'): aug.append(w[:-2])
        return q + " " + " ".join(aug)

class AdvancedVectorStore:
    def __init__(self, embedding_model):
        self.emb_model = embedding_model
        self.dim = embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.embeddings = None
        self.chunks = []
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.expander = QueryExpander()
    
    def build(self, chunks):
        texts = [c["text"] for c in chunks]
        print(f"Building indexes for {len(texts)} chunks")
        E = self.emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256).astype("float32")
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(E)
        self.embeddings = E
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=6000, stop_words='english', ngram_range=(1,2), min_df=1, max_df=0.95)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.chunks = chunks
        print("Indexing complete")
    
    def search(self, query, top_k=5, use_expansion=True, rerank=True):
        search_start = time.time()
        if not self.chunks:
            return [], time.time() - search_start
        expanded = self.expander.expand(query) if use_expansion else query
        cands = self._retrieve_multi(query, expanded, top_k*3)
        if rerank and len(cands)>top_k:
            cands = self._rerank(query, cands, top_k*2)
        final = self._diversify(cands[:top_k*2], top_k)
        search_time = time.time() - search_start
        return final[:top_k], search_time
    
    def _retrieve_multi(self, orig_q, expanded_q, top_k):
        res={}
        qv = self.emb_model.encode([orig_q], normalize_embeddings=True).astype("float32")
        sims, idxs = self.index.search(qv, min(top_k*2, len(self.chunks)))
        for i,s in zip(idxs[0], sims[0]):
            if i < len(self.chunks):
                res[int(i)] = res.get(int(i),0) + float(s)*0.4
        bm = self.bm25.get_scores(expanded_q.lower().split())
        bm_top = np.argsort(bm)[::-1][:min(top_k*2, len(self.chunks))]
        mbm = max(bm) if len(bm)>0 else 1
        for i in bm_top:
            res[i] = res.get(i,0) + (bm[i]/mbm if mbm>0 else 0)*0.3
        qtf = self.tfidf_vectorizer.transform([expanded_q])
        from sklearn.metrics.pairwise import cosine_similarity
        tf = cosine_similarity(qtf, self.tfidf_matrix).flatten()
        tf_top = np.argsort(tf)[::-1][:min(top_k*2, len(self.chunks))]
        for i in tf_top:
            res[i] = res.get(i,0) + float(tf[i])*0.3
        ranked = sorted(res.items(), key=lambda x:x[1], reverse=True)
        out=[]
        for idx,score in ranked[:top_k]:
            t = self.chunks[idx]["text"]
            score += MortgageHeuristics.domain_score(t)  # domain boost/penalty
            out.append({"text": t, "metadata": self.chunks[idx]["metadata"].copy(), "relevance": round(score,4), "chunk_idx": idx})
        out.sort(key=lambda x:x["relevance"], reverse=True)
        return out
    
    def _rerank(self, query, cands, top_k):
        q_terms = set(query.lower().split())
        def score(c):
            text = c["text"].lower()
            txt_terms = set(text.split())
            overlap = len(q_terms & txt_terms) / max(1,len(q_terms))
            pos_scores = [1-(text.find(t)/max(1,len(text))) for t in q_terms if t in text]
            pos = np.mean(pos_scores) if pos_scores else 0
            return c["relevance"]*0.6 + overlap*0.25 + pos*0.15 + MortgageHeuristics.domain_score(text)*0.5
        rer = sorted(cands, key=score, reverse=True)
        return rer[:top_k]
    
    def _diversify(self, cands, top_k):
        if len(cands)<=top_k: return cands
        sel = [cands[0]]
        rem = cands[1:]
        while len(sel)<top_k and rem:
            best = None
            best_score = -1
            for c in rem:
                sims = []
                for s in sel:
                    w1 = set(c["text"].lower().split())
                    w2 = set(s["text"].lower().split())
                    j = len(w1&w2)/max(1,len(w1|w2))
                    sims.append(j)
                div = 1-max(sims) if sims else 1
                sc = c["relevance"]*0.7 + div*0.3
                if sc>best_score: best_score=sc; best=c
            sel.append(best)
            rem.remove(best)
        return sel

class EnhancedQAModel:
    def __init__(self):
        m = "deepset/roberta-base-squad2"
        self.pipe = pipeline("question-answering", model=m, tokenizer=m, device=0 if torch.cuda.is_available() else -1)
    
    def answer(self, question, hits):
        qa_start = time.time()
        if not hits:
            return {"found": False, "confidence": 0.0, "processing_time": 0.0}
        
        best = None
        score = 0.0
        for h in hits[:3]:
            try:
                # Increase context length and improve processing
                context = h["text"][:4000]  # Increased from 3000 for more complete answers
                out = self.pipe({"question":question, "context":context})
                ans = (out.get("answer") or "").strip()
                sc = float(out.get("score",0.0))

                # Better answer validation
                if sc>score and ans and len(ans.split()) >= 2:  # Minimum 2 words
                    processed_answer = self._process_answer(ans, context, out.get("start",-1), out.get("end",-1))
                    if processed_answer:
                        best = {"answer": processed_answer, "score": sc}
                        score = sc
            except Exception as e:
                print(f"QA processing error: {e}")
                pass

        qa_time = time.time() - qa_start
        # Stricter confidence threshold
        if not best or score < 0.3:  # Increased from 0.2
            return {"found": False, "confidence": score, "processing_time": qa_time}
        return {"found": True, "answer": best["answer"], "score": score, "confidence": min(1.0, score*1.2), "processing_time": qa_time}

    def _process_answer(self, ans, ctx, start, end):
        """Better answer processing with complete sentence extraction"""
        if start == -1 or end == -1:
            start = ctx.lower().find(ans.lower())
            if start == -1:
                return None
            end = start + len(ans)

        # Find sentence boundaries
        sent_start = start
        sent_end = end

        # Expand backwards to sentence start
        while sent_start > 0 and ctx[sent_start-1] not in '.!?\n':
            sent_start -= 1
        if sent_start > 0 and ctx[sent_start-1] in '.!?':
            sent_start += 1

        # Expand forwards to sentence end
        while sent_end < len(ctx) and ctx[sent_end] not in '.!?\n':
            sent_end += 1
        if sent_end < len(ctx) and ctx[sent_end] in '.!?':
            sent_end += 1

        sentence = ctx[sent_start:sent_end].strip()

        # Clean up the sentence
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.strip()

        # Validate the sentence contains the answer
        if ans.lower() not in sentence.lower():
            return ans.strip() + "." if not sentence.endswith(('.', '!', '?')) else ans.strip()

        if not sentence.endswith(('.', '!', '?')) and len(sentence.split()) > 3:
            sentence += "."

        return sentence
