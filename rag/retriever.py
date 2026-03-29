"""
rag/retriever.py

Builds and queries a FAISS vector index over the longevity paper corpus.
Uses sentence-transformers for local (free) embeddings by default.
Can switch to OpenAI embeddings via USE_LOCAL_EMBEDDINGS=false in .env
"""

import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class LongevityRetriever:
    """
    Vector-store retriever over longevity/aging research papers.

    Usage:
        retriever = LongevityRetriever()
        retriever.build()
        results = retriever.search("rapamycin lifespan extension", k=3)
    """

    INDEX_PATH = "models/saved/faiss.index"
    META_PATH  = "models/saved/faiss_meta.json"

    def __init__(self):
        self.use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
        self._embedder  = None
        self._index     = None
        self._papers    = []

    # ── Embedding ────────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder

        if self.use_local:
            from sentence_transformers import SentenceTransformer
            print("Loading local embedding model (all-MiniLM-L6-v2)...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            from openai import OpenAI
            self._embedder = OpenAI()
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        if self.use_local:
            return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            resp = embedder.embeddings.create(model="text-embedding-3-small", input=texts)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            # L2-normalise for cosine similarity via inner product
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / (norms + 1e-10)

    # ── Build index ───────────────────────────────────────────────────────────

    def build(self, force: bool = False):
        if not force and Path(self.INDEX_PATH).exists():
            print("Loading existing FAISS index...")
            return self.load()

        import faiss
        from rag.corpus import get_all_papers, format_for_embedding

        self._papers = get_all_papers()
        texts = [format_for_embedding(p) for p in self._papers]

        print(f"Embedding {len(texts)} papers...")
        vecs = self._embed(texts).astype(np.float32)

        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalisation
        self._index.add(vecs)

        os.makedirs("models/saved", exist_ok=True)
        faiss.write_index(self._index, self.INDEX_PATH)
        with open(self.META_PATH, "w") as f:
            json.dump(self._papers, f)

        print(f"✓ FAISS index built: {self._index.ntotal} vectors, dim={dim}")
        return self

    def load(self):
        import faiss
        self._index = faiss.read_index(self.INDEX_PATH)
        with open(self.META_PATH) as f:
            self._papers = json.load(f)
        return self

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return top-k papers most relevant to query."""
        if self._index is None:
            self.load()

        q_vec = self._embed([query]).astype(np.float32)
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            paper = dict(self._papers[idx])
            paper["relevance_score"] = float(score)
            results.append(paper)
        return results

    def format_context(self, results: list[dict]) -> str:
        """Format search results into a context block for LLM prompting."""
        lines = []
        for i, p in enumerate(results, 1):
            lines.append(
                f"[{i}] {p['title']} — {p['authors']} ({p['journal']}, {p['year']})\n"
                f"    {p['abstract'][:400]}..."
            )
        return "\n\n".join(lines)

    # ── Hallucination guard ───────────────────────────────────────────────────

    def verify_claim(self, claim: str, threshold: float = 0.35) -> dict:
        """
        Check if a claim is grounded in the retrieval corpus.
        Returns {grounded: bool, supporting_papers: list, max_score: float}
        """
        results = self.search(claim, k=3)
        max_score = results[0]["relevance_score"] if results else 0.0
        return {
            "grounded": max_score >= threshold,
            "max_score": max_score,
            "supporting_papers": [r["title"] for r in results if r["relevance_score"] >= threshold],
        }
