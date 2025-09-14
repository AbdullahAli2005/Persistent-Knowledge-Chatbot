import os
import json
import faiss
import numpy as np
from typing import List, Dict
import google.generativeai as genai

CHAT_MEMORY_PATH = "data/chat_memory.json"

# EMBED_DIM: pick a dimension; we'll infer from embedding response
# but assume embed_content returns vector of fixed size.
# We'll set EMBED_DIM when we get the first embedding.
EMBED_DIM = None

def _text_to_embedding(text: str) -> List[float]:
    """Convert text into embedding using Gemini embed_content."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    # response["embedding"] or response.embeddings may vary
    emb = None
    if "embedding" in response:
        emb = response["embedding"]
    elif hasattr(response, "embeddings"):
        # assume response.embeddings is list of embedding objects
        emb = response.embeddings[0].values
    else:
        raise RuntimeError("Unexpected embed_content response shape: " + str(response))
    return emb

class MemoryManager:
    def __init__(self):
        self.index = None
        self.vectors = []
        self.metadata = []
        # ensure directory exists
        os.makedirs(os.path.dirname(CHAT_MEMORY_PATH), exist_ok=True)
        self.reindex_from_json()

    def _read_json(self) -> List[Dict]:
        if os.path.exists(CHAT_MEMORY_PATH):
            try:
                with open(CHAT_MEMORY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _write_json(self, data: List[Dict]):
        os.makedirs(os.path.dirname(CHAT_MEMORY_PATH), exist_ok=True)
        with open(CHAT_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _add_to_chroma(self, idx: int, question: str, answer: str, embedding=None):
        """Add vector + metadata into FAISS index (embedding provided or computed)."""
        if embedding is None:
            doc_text = f"Q: {question}\nA: {answer}"
            embedding = _text_to_embedding(doc_text)

        emb = np.array([embedding], dtype="float32")

        global EMBED_DIM
        if EMBED_DIM is None:
            EMBED_DIM = len(embedding)

        if self.index is None:
            self.index = faiss.IndexFlatL2(EMBED_DIM)

        self.index.add(emb)
        self.vectors.append(emb)
        self.metadata.append({"question": question, "answer": answer})

    def reindex_from_json(self):
        """Rebuild FAISS index from JSON with cached embeddings if available."""
        self.index = None
        self.vectors = []
        self.metadata = []

        pairs = self._read_json()
        for idx, p in enumerate(pairs):
            q = p.get("question", "")
            a = p.get("answer", "")
            emb = p.get("embedding")
            if emb is not None:
                # use cached embedding
                self._add_to_chroma(idx, q, a, emb)
            else:
                self._add_to_chroma(idx, q, a, None)

    def append_pair(self, question: str, answer: str):
        """Append new Q/A pair, compute embedding, store both."""
        doc_text = f"Q: {question}\nA: {answer}"
        embedding = _text_to_embedding(doc_text)

        pairs = self._read_json()
        pairs.append({
            "question": question,
            "answer": answer,
            "embedding": embedding
        })
        self._write_json(pairs)

        self._add_to_chroma(len(pairs) - 1, question, answer, embedding)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for most relevant past Q/A pairs."""
        if self.index is None or len(self.metadata) == 0:
            # fallback: return last k from JSON
            pairs = self._read_json()[-k:]
            return [
                {
                    "question": p.get("question"),
                    "answer": p.get("answer"),
                    "score": None
                }
                for p in reversed(pairs)
            ]

        query_emb = np.array([_text_to_embedding(query)], dtype="float32")
        distances, indices = self.index.search(query_emb, min(k, len(self.metadata)))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "question": self.metadata[idx]["question"],
                    "answer": self.metadata[idx]["answer"],
                    "score": float(distances[0][i])
                })
        return results
