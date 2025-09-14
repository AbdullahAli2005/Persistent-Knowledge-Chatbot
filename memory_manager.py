import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load env
load_dotenv()

CHAT_MEMORY_PATH = os.getenv("CHAT_MEMORY_PATH", "data/chat_memory.json")
EMBED_DIM = 768  # Gemini embedding dimension

# Configure Gemini Embedding
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embed_model = "models/embedding-001"


def _text_to_embedding(text: str) -> np.ndarray:
    """Convert text into a Gemini embedding vector."""
    resp = genai.embed_content(model=embed_model, content=text)
    return np.array(resp["embedding"], dtype="float32")


class MemoryManager:
    def __init__(self):
        os.makedirs(os.path.dirname(CHAT_MEMORY_PATH), exist_ok=True)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

        # Load and reindex stored Q&A
        self.reindex_from_json()

    def _read_json(self) -> List[Dict[str, str]]:
        if not os.path.exists(CHAT_MEMORY_PATH):
            return []
        try:
            with open(CHAT_MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_json(self, data: List[Dict[str, str]]):
        with open(CHAT_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _add_to_chroma(self, idx: int, question: str, answer: str):
        """Index a Q&A pair into FAISS (kept function name for compatibility)."""
        doc_text = f"Q: {question}\nA: {answer}"
        emb = np.array([_text_to_embedding(doc_text)], dtype="float32")
        self.index.add(emb)
        self.vectors.append(emb)
        self.metadata.append({"question": question, "answer": answer})

    def append_pair(self, question: str, answer: str):
        """Save new Q&A pair into JSON and FAISS."""
        pairs = self._read_json()
        pairs.append({"question": question, "answer": answer})
        self._write_json(pairs)
        self._add_to_chroma(len(pairs) - 1, question, answer)

    def reindex_from_json(self):
        """Rebuild FAISS index from stored JSON Q&A."""
        pairs = self._read_json()
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.vectors = []
        self.metadata = []

        for idx, p in enumerate(pairs):
            q, a = p.get("question", ""), p.get("answer", "")
            self._add_to_chroma(idx, q, a)

    def query_similar(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Find most similar Q&A pairs using FAISS nearest neighbors."""
        emb = np.array([_text_to_embedding(query_text)], dtype="float32")

        if self.index.ntotal == 0:
            # If no data, fallback to last N Q&A from JSON
            pairs = self._read_json()[-n_results:]
            return [
                {
                    "question": p.get("question"),
                    "answer": p.get("answer"),
                    "document": f"Q: {p.get('question')}\nA: {p.get('answer')}",
                }
                for p in reversed(pairs)
            ]

        distances, indices = self.index.search(emb, n_results)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(
                    {
                        "document": f"Q: {meta['question']}\nA: {meta['answer']}",
                        "question": meta["question"],
                        "answer": meta["answer"],
                        "distance": float(dist),
                        "id": idx,
                    }
                )
        return results
