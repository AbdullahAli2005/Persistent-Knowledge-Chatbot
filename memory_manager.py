import os
import json
import faiss
import numpy as np
from typing import List, Dict
from google import genai


CHAT_MEMORY_PATH = "data/chat_memory.json"

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def _text_to_embedding(text: str) -> List[float]:
    """Convert text into embedding using Gemini 1.5 Flash."""
    response = client.models.embed_content(
        model="models/embedding-001",
        input=text,
    )
    return response.embeddings[0].values


class MemoryManager:
    def __init__(self):
        self.index = None
        self.vectors = []
        self.metadata = []
        self.reindex_from_json()

    def _read_json(self) -> List[Dict]:
        if os.path.exists(CHAT_MEMORY_PATH):
            with open(CHAT_MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _write_json(self, data: List[Dict]):
        os.makedirs(os.path.dirname(CHAT_MEMORY_PATH), exist_ok=True)
        with open(CHAT_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _add_to_chroma(self, idx: int, question: str, answer: str, embedding=None):
        """Add vector + metadata into FAISS index."""
        if embedding is None:
            doc_text = f"Q: {question}\nA: {answer}"
            embedding = _text_to_embedding(doc_text)

        emb = np.array([embedding], dtype="float32")

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embedding))

        self.index.add(emb)
        self.vectors.append(emb)
        self.metadata.append({"question": question, "answer": answer})

    def reindex_from_json(self):
        """Rebuild FAISS index from JSON memory file."""
        self.index = None
        self.vectors = []
        self.metadata = []

        pairs = self._read_json()
        for idx, p in enumerate(pairs):
            q = p.get("question", "")
            a = p.get("answer", "")
            emb = p.get("embedding")
            if emb is not None:
                emb = np.array(emb, dtype="float32")
            self._add_to_chroma(idx, q, a, emb)

    def append_pair(self, question: str, answer: str):
        """Append new Q/A pair and store embedding."""
        doc_text = f"Q: {question}\nA: {answer}"
        emb = _text_to_embedding(doc_text)

        pairs = self._read_json()
        pairs.append({
            "question": question,
            "answer": answer,
            "embedding": emb
        })
        self._write_json(pairs)

        self._add_to_chroma(len(pairs) - 1, question, answer, emb)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for most relevant past Q/A pairs."""
        if self.index is None:
            return []

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
