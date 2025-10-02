import os
import json
import faiss
import numpy as np
from typing import List, Dict
import google.generativeai as genai

CHAT_MEMORY_PATH = "data/chat_memory.json"
EMBED_DIM = 768  # Set a default dimension for Gemini embeddings

def _text_to_embedding(text: str) -> List[float]:
    """Convert text into embedding using Gemini embed_content."""
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        # Handle the response structure
        if hasattr(response, 'embedding'):
            return response.embedding
        elif 'embedding' in response:
            return response['embedding']
        else:
            # Fallback: return the first element if it's nested
            return response['embeddings'][0] if 'embeddings' in response else []
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * EMBED_DIM  # Return zero vector as fallback

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

    def _add_to_index(self, embedding: List[float], question: str, answer: str):
        """Add vector + metadata to FAISS index."""
        if not embedding:
            return
            
        emb_array = np.array([embedding], dtype="float32")
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embedding))
        
        self.index.add(emb_array)
        self.metadata.append({"question": question, "answer": answer})

    def reindex_from_json(self):
        """Rebuild FAISS index from JSON."""
        self.index = None
        self.metadata = []

        pairs = self._read_json()
        for p in pairs:
            q = p.get("question", "")
            a = p.get("answer", "")
            emb = p.get("embedding")
            if emb and len(emb) > 0:
                self._add_to_index(emb, q, a)
            else:
                # Compute embedding if not present
                doc_text = f"Q: {q}\nA: {a}"
                new_emb = _text_to_embedding(doc_text)
                self._add_to_index(new_emb, q, a)

    def append_pair(self, question: str, answer: str):
        """Append new Q/A pair with embedding."""
        doc_text = f"Q: {question}\nA: {answer}"
        embedding = _text_to_embedding(doc_text)

        pairs = self._read_json()
        pairs.append({
            "question": question,
            "answer": answer,
            "embedding": embedding
        })
        self._write_json(pairs)
        
        self._add_to_index(embedding, question, answer)

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

        query_emb = _text_to_embedding(query)
        if not query_emb:
            return []
            
        query_array = np.array([query_emb], dtype="float32")
        distances, indices = self.index.search(query_array, min(k, len(self.metadata)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "question": self.metadata[idx]["question"],
                    "answer": self.metadata[idx]["answer"],
                    "score": float(distances[0][i])
                })
        return results