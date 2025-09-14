import os
import json
import hashlib
import math
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception as e:
    raise ImportError("ChromaDB is required for this module. Please install it via 'pip install chromadb'") from e

# Config defaults
DEFAULT_CHAT_MEMORY_PATH = os.getenv("CHAT_MEMORY_PATH", "data/chat_memory.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")  # default to a folder

# Embedding dimension used for pseudo embeddings
EMBED_DIM = 128


def _text_to_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    """
    Produce a deterministic pseudo-embedding from text using hashing.
    Not semantically strong like real embeddings, but deterministic and
    adequate for demonstration and reindexing.
    """
    sha = hashlib.sha256(text.encode("utf-8")).digest()
    md5 = hashlib.md5(text.encode("utf-8")).digest()
    combined = sha + md5  # 48 bytes
    floats = []
    for i in range(dim):
        b = combined[i % len(combined)]
        val = (b / 255.0) * 2.0 - 1.0  # normalize -1..1
        floats.append(math.sin(val * (i + 1)))
    return floats


class MemoryManager:
    def __init__(
        self,
        chat_memory_path: str = DEFAULT_CHAT_MEMORY_PATH,
        chroma_persist_dir: Optional[str] = CHROMA_PERSIST_DIR,
    ):
        self.chat_memory_path = chat_memory_path
        self.persist_dir = chroma_persist_dir or "data/chroma"

        # ensure dirs
        self.ensure_data_dir()
        os.makedirs(self.persist_dir, exist_ok=True)

        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Default embedding function (can be replaced with OpenAI/Gemini later)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection_name = "chat_memory"
        self.collection = self._get_or_create_collection()

        # On init, reindex JSON
        self.reindex_from_json()

    def ensure_data_dir(self):
        folder = os.path.dirname(self.chat_memory_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def _get_or_create_collection(self):
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
            )
        except Exception:
            # last resort
            return self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
            )

    def _read_json(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.chat_memory_path):
            return []
        try:
            with open(self.chat_memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write_json(self, arr: List[Dict[str, str]]):
        with open(self.chat_memory_path, "w", encoding="utf-8") as f:
            json.dump(arr, f, indent=2, ensure_ascii=False)

    def get_all_pairs(self) -> List[Dict[str, str]]:
        """Returns list of {"question": str, "answer": str}"""
        return self._read_json()

    def append_pair(self, question: str, answer: str):
        arr = self._read_json()
        arr.append({"question": question, "answer": answer})
        self._write_json(arr)
        # reindex only new item
        self._add_to_chroma(len(arr) - 1, question, answer)

    def reindex_from_json(self):
        """Rebuild the entire collection from JSON file"""
        pairs = self._read_json()
        try:
            # clear existing collection
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
        except Exception:
            pass

        if not pairs:
            return

        ids, documents, metadatas, embeddings = [], [], [], []
        for idx, p in enumerate(pairs):
            q, a = p.get("question", ""), p.get("answer", "")
            doc_text = f"Q: {q}\nA: {a}"
            ids.append(str(idx))
            documents.append(doc_text)
            metadatas.append({"question": q, "answer": a})
            embeddings.append(_text_to_embedding(doc_text))

        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as e:
            print("Warning: Chroma add failed during reindex.", e)

    def _add_to_chroma(self, idx: int, question: str, answer: str):
        doc_text = f"Q: {question}\nA: {answer}"
        try:
            self.collection.add(
                ids=[str(idx)],
                documents=[doc_text],
                metadatas=[{"question": question, "answer": answer}],
                embeddings=[_text_to_embedding(doc_text)],
            )
        except Exception as e:
            print("Warning: Chroma add for single item failed.", e)

    def query_similar(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Query Chroma for top-n similar Q+A entries.
        Fallback: return last n from JSON.
        """
        emb = _text_to_embedding(query_text)
        try:
            res = self.collection.query(
                query_embeddings=[emb],
                n_results=n_results,
                include=["metadatas", "documents", "distances", "ids"],
            )
            results = []
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0] if "distances" in res else []
            ids = res.get("ids", [[]])[0] if "ids" in res else []

            for doc, meta, dist, id_ in zip(docs, metas, dists, ids):
                results.append(
                    {
                        "document": doc,
                        "question": meta.get("question") if meta else None,
                        "answer": meta.get("answer") if meta else None,
                        "distance": dist,
                        "id": id_,
                    }
                )
            return results
        except Exception:
            pairs = self._read_json()[-n_results:]
            return [
                {
                    "question": p.get("question"),
                    "answer": p.get("answer"),
                    "document": f"Q: {p.get('question')}\nA: {p.get('answer')}",
                }
                for p in reversed(pairs)
            ]
