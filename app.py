# import os
# import streamlit as st
# from dotenv import load_dotenv
# from memory_manager import MemoryManager
# import google.generativeai as genai
# from typing import List, Optional
# from langchain.llms.base import LLM
# from pydantic import Field, PrivateAttr

# # Load env
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Set GOOGLE_API_KEY in your .env file before running the app.")

# genai.configure(api_key=GOOGLE_API_KEY)


# # --- Fixed GeminiLLM ---
# # class GeminiLLM(LLM):
#     # model: str = Field(default="gemini-1.5-flash", description="Gemini model name")
#     # _client: genai.GenerativeModel = PrivateAttr()

#     # def __init__(self, **data):
#     #     super().__init__(**data)
#     #     # Initialize the Gemini client
#     #     self._client = genai.GenerativeModel(self.model)

#     # @property
#     # def _llm_type(self) -> str:
#     #     return "gemini-llm"

#     # def _call(self, prompt: str, stop=None):
#     #     try:
#     #         response = self._client.generate_content(prompt)
#     #         return response.text
#     #     except Exception as e:
#     #         return f"Error from Gemini API: {e}"

# class GeminiLLM(LLM):
#     model: str = Field(default="gemini-1.5-flash", description="Gemini model name")
#     _client: genai.GenerativeModel = PrivateAttr()

#     def __init__(self, **data):
#         super().__init__(**data)
#         try:
#             self._client = genai.GenerativeModel(self.model)
#         except Exception as e:
#             raise RuntimeError(f"Failed to init Gemini model {self.model}: {e}")

#     @property
#     def _llm_type(self) -> str:
#         return "gemini-llm"

#     def _call(self, prompt: str, stop=None):
#         try:
#             response = self._client.generate_content(prompt)

#             # Gemini returns candidates, not always response.text
#             if hasattr(response, "text"):
#                 return response.text
#             elif hasattr(response, "candidates"):
#                 return response.candidates[0].content.parts[0].text
#             else:
#                 return str(response)

#         except Exception as e:
#             return f"Error from Gemini API: {e}"


# # --- Streamlit UI ---
# st.set_page_config(page_title="Persistent Knowledge Base Chatbot", page_icon="🤖", layout="wide")
# st.title("🤖 Persistent Knowledge Base Chatbot")

# with st.spinner("Initializing memory and FAISS..."):
#     mem = MemoryManager()

# st.sidebar.header("Chat Memory")
# history = mem._read_json()
# if history:
#     for i, pair in enumerate(history[::-1]):
#         st.sidebar.markdown(f"**Q:** {pair['question']}")
#         st.sidebar.markdown(f"**A:** {pair['answer']}")
#         st.sidebar.markdown("---")
# else:
#     st.sidebar.info("No chat memory yet. Ask a question to start!")

# st.header("Ask anything (Gemini-powered)")
# query = st.text_area("Your question", height=120, placeholder="Type your question here...")

# col1, col2 = st.columns([1, 1])
# with col1:
#     btn = st.button("Send")
# with col2:
#     clear = st.button("Clear chat memory (delete JSON & FAISS index)")

# if clear:
#     mem._write_json([])
#     mem.reindex_from_json()

# if btn and query:
#     context = mem.search(query, k=3)
#     context_parts = []
#     for c in context:
#         q = c.get("question") or ""
#         a = c.get("answer") or ""
#         if q and a:
#             context_parts.append(f"Q: {q}\nA: {a}")
#     context_text = "\n\n".join(context_parts)

#     system_instruction = (
#         "You are a helpful assistant. Use any provided chat history context (previous Q/A pairs) "
#         "to help answer the user's question. If no context is helpful, answer based on your knowledge."
#     )

#     prompt = system_instruction + "\n\n"
#     if context_text:
#         prompt += "Relevant chat memory (previous Q/A pairs):\n" + context_text + "\n\n"
#     prompt += "User question:\n" + query + "\n\nAnswer:"

#     llm = GeminiLLM()
#     with st.spinner("Generating answer from Gemini..."):
#         try:
#             answer = llm(prompt)
#         except Exception as e:
#             st.error("Error calling Gemini LLM: " + str(e))
#             answer = "Error generating answer."

#     mem.append_pair(query, answer)

#     st.markdown("### Answer")
#     st.write(answer)

#     st.markdown("---")
#     st.markdown("### Used context (top results from memory)")
#     if context_parts:
#         for i, c in enumerate(context_parts):
#             st.markdown(f"**Context #{i+1}**")
#             st.text(c)
#     else:
#         st.info("No relevant context found in memory.")
# else:
#     st.info("Type a question and press Send to receive an answer from Gemini.")
import os
import json
import faiss
import numpy as np
from typing import List, Dict
import google.generativeai as genai

CHAT_MEMORY_PATH = "data/chat_memory.json"
EMBED_DIM = 768  # Default dimension for Gemini embeddings

def _text_to_embedding(text: str) -> List[float]:
    """Convert text into embedding using Gemini embed_content."""
    try:
        # Clean the text to avoid API issues
        clean_text = text.strip()
        if not clean_text:
            return [0.0] * EMBED_DIM
            
        response = genai.embed_content(
            model="models/embedding-001",
            content=clean_text,
            task_type="retrieval_document"
        )
        
        # Handle different response formats
        if hasattr(response, 'embedding'):
            return response.embedding
        elif isinstance(response, dict) and 'embedding' in response:
            return response['embedding']
        elif hasattr(response, 'embeddings') and response.embeddings:
            return response.embeddings[0].values
        else:
            # Try to extract embedding from the response
            return getattr(response, 'embedding', [0.0] * EMBED_DIM)
            
    except Exception as e:
        print(f"Embedding error for text '{text[:50]}...': {e}")
        return [0.0] * EMBED_DIM

class MemoryManager:
    def __init__(self):
        self.index = None
        self.metadata = []
        # ensure directory exists
        os.makedirs(os.path.dirname(CHAT_MEMORY_PATH), exist_ok=True)
        self.reindex_from_json()

    def _read_json(self) -> List[Dict]:
        if os.path.exists(CHAT_MEMORY_PATH):
            try:
                with open(CHAT_MEMORY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception as e:
                print(f"Error reading JSON: {e}")
                return []
        return []

    def _write_json(self, data: List[Dict]):
        try:
            os.makedirs(os.path.dirname(CHAT_MEMORY_PATH), exist_ok=True)
            with open(CHAT_MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing JSON: {e}")

    def _add_to_index(self, embedding: List[float], question: str, answer: str):
        """Add vector + metadata to FAISS index."""
        if not embedding or len(embedding) == 0:
            print("Empty embedding, skipping index addition")
            return
            
        try:
            emb_array = np.array([embedding], dtype="float32")
            
            if self.index is None:
                # Initialize index with correct dimension
                actual_dim = len(embedding)
                self.index = faiss.IndexFlatL2(actual_dim)
            
            self.index.add(emb_array)
            self.metadata.append({"question": question, "answer": answer})
        except Exception as e:
            print(f"Error adding to index: {e}")

    def reindex_from_json(self):
        """Rebuild FAISS index from JSON."""
        self.index = None
        self.metadata = []

        pairs = self._read_json()
        if not pairs:
            return
            
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
        if not question.strip() or not answer.strip():
            print("Empty question or answer, skipping")
            return
            
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
        if not query.strip():
            return []
            
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

        try:
            query_emb = _text_to_embedding(query)
            if not query_emb or len(query_emb) == 0:
                return []
                
            query_array = np.array([query_emb], dtype="float32")
            k_actual = min(k, len(self.metadata))
            distances, indices = self.index.search(query_array, k_actual)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.metadata):
                    results.append({
                        "question": self.metadata[idx]["question"],
                        "answer": self.metadata[idx]["answer"],
                        "score": float(distances[0][i])
                    })
            return results
        except Exception as e:
            print(f"Search error: {e}")
            # Fallback to recent pairs
            pairs = self._read_json()[-k:]
            return [
                {
                    "question": p.get("question"),
                    "answer": p.get("answer"),
                    "score": None
                }
                for p in reversed(pairs)
            ]