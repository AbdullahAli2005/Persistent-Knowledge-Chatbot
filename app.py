import os
import streamlit as st
from dotenv import load_dotenv
from memory_manager import MemoryManager
import google.generativeai as genai

# Load env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your .env file before running the app.")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Simplified GeminiLLM ---
class GeminiLLM:
    def __init__(self, model="gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model)
    
    def __call__(self, prompt: str):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error from Gemini API: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Persistent Knowledge Base Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Persistent Knowledge Base Chatbot")

# Initialize memory manager
with st.spinner("Initializing memory and FAISS..."):
    mem = MemoryManager()

st.sidebar.header("Chat Memory")
history = mem._read_json()
if history:
    for i, pair in enumerate(history[::-1]):
        st.sidebar.markdown(f"**Q:** {pair['question']}")
        st.sidebar.markdown(f"**A:** {pair['answer']}")
        st.sidebar.markdown("---")
else:
    st.sidebar.info("No chat memory yet. Ask a question to start!")

st.header("Ask anything (Gemini-powered)")
query = st.text_area("Your question", height=120, placeholder="Type your question here...")

col1, col2 = st.columns([1, 1])
with col1:
    btn = st.button("Send")
with col2:
    clear = st.button("Clear chat memory (delete JSON & FAISS index)")

if clear:
    mem._write_json([])
    mem.reindex_from_json()
    st.success("Chat memory cleared!")
    st.experimental_rerun()

if btn and query:
    # Initialize LLM here to avoid early initialization issues
    llm = GeminiLLM()
    
    context = mem.search(query, k=3)
    context_parts = []
    for c in context:
        q = c.get("question") or ""
        a = c.get("answer") or ""
        if q and a:
            context_parts.append(f"Q: {q}\nA: {a}")
    context_text = "\n\n".join(context_parts)

    system_instruction = (
        "You are a helpful assistant. Use any provided chat history context (previous Q/A pairs) "
        "to help answer the user's question. If no context is helpful, answer based on your knowledge."
    )

    prompt = system_instruction + "\n\n"
    if context_text:
        prompt += "Relevant chat memory (previous Q/A pairs):\n" + context_text + "\n\n"
    prompt += "User question:\n" + query + "\n\nAnswer:"

    with st.spinner("Generating answer from Gemini..."):
        try:
            answer = llm(prompt)
        except Exception as e:
            st.error("Error calling Gemini LLM: " + str(e))
            answer = "Error generating answer."

    mem.append_pair(query, answer)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("---")
    st.markdown("### Used context (top results from memory)")
    if context_parts:
        for i, c in enumerate(context_parts):
            st.markdown(f"**Context #{i+1}**")
            st.text(c)
    else:
        st.info("No relevant context found in memory.")
else:
    st.info("Type a question and press Send to receive an answer from Gemini.")