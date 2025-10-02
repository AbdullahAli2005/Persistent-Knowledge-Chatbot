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



# import os
# import streamlit as st
# from dotenv import load_dotenv
# from memory_manager import MemoryManager
# import google.generativeai as genai

# # Load env
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Set GOOGLE_API_KEY in your .env file before running the app.")

# genai.configure(api_key=GOOGLE_API_KEY)

# # --- Simplified GeminiLLM ---
# class GeminiLLM:
#     def __init__(self, model="gemini-1.5-flash"):
#         self.model = genai.GenerativeModel(model)
    
#     def __call__(self, prompt: str):
#         try:
#             response = self.model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             return f"Error from Gemini API: {e}"

# # --- Streamlit UI ---
# st.set_page_config(page_title="Persistent Knowledge Base Chatbot", page_icon="🤖", layout="wide")
# st.title("🤖 Persistent Knowledge Base Chatbot")

# # Initialize memory manager
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
#     st.success("Chat memory cleared!")
#     st.experimental_rerun()

# if btn and query:
#     # Initialize LLM here to avoid early initialization issues
#     llm = GeminiLLM()
    
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
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    st.success("✅ Gemini API configured successfully")
except Exception as e:
    st.error(f"❌ Error configuring Gemini: {e}")
    st.stop()

# Discover available models
@st.cache_data
def get_available_models():
    """Get list of available models that support generateContent"""
    try:
        available_models = []
        models = genai.list_models()
        
        for model in models:
            # Check if model supports generateContent
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        return available_models
    except Exception as e:
        st.error(f"Error listing models: {e}")
        return []

def get_gemini_response(prompt):
    """Get response from Gemini using available models"""
    try:
        available_models = get_available_models()
        
        if not available_models:
            return "Error: No available models found that support content generation"
        
        # Try models in order of preference
        preferred_models = [
            "gemini-1.5-flash-001",
            "gemini-1.0-pro-001", 
            "gemini-1.0-pro",
            "models/gemini-pro"
        ]
        
        # Filter to only available models
        models_to_try = [model for model in preferred_models if model in available_models]
        
        # If none of our preferred models are available, use the first available one
        if not models_to_try and available_models:
            models_to_try = [available_models[0]]
        
        if not models_to_try:
            return "Error: No suitable models available"
        
        # Try each model until one works
        last_error = ""
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response.text:
                    st.success(f"✅ Using model: {model_name}")
                    return response.text
            except Exception as e:
                last_error = str(e)
                continue
        
        return f"Error from Gemini: {last_error}"
        
    except Exception as e:
        return f"Error: {str(e)}"

# Simple memory management
CHAT_MEMORY_PATH = "chat_memory.json"

def load_chat_memory():
    """Load chat history from JSON file"""
    try:
        if os.path.exists(CHAT_MEMORY_PATH):
            with open(CHAT_MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading chat memory: {e}")
        return []

def save_chat_memory(memory):
    """Save chat history to JSON file"""
    try:
        with open(CHAT_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving chat memory: {e}")

# Streamlit UI
st.set_page_config(
    page_title="Persistent Knowledge Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Persistent Knowledge Chatbot")
st.markdown("---")

# Display available models for debugging
try:
    available_models = get_available_models()
    with st.expander("🔧 Debug Info - Available Models"):
        if available_models:
            st.write("Models supporting generateContent:")
            for model in available_models:
                st.write(f"- {model}")
        else:
            st.write("No models found or error fetching models")
except Exception as e:
    st.error(f"Error getting model info: {e}")

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = load_chat_memory()

# Sidebar for chat history
st.sidebar.header("💬 Chat History")

if st.session_state.memory:
    for i, chat in enumerate(st.session_state.memory[-10:]):  # Show last 10
        with st.sidebar.expander(f"Chat {i+1}", expanded=False):
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
else:
    st.sidebar.info("No chat history yet. Start a conversation!")

# Clear chat button
if st.sidebar.button("🗑️ Clear All Chat History"):
    st.session_state.memory = []
    save_chat_memory([])
    st.sidebar.success("Chat history cleared!")
    st.rerun()

# Main chat interface
st.header("💭 Ask me anything")

question = st.text_area(
    "Your question:",
    height=100,
    placeholder="Type your question here...",
    key="question_input"
)

col1, col2 = st.columns([1, 1])

with col1:
    send_btn = st.button("🚀 Send", type="primary", use_container_width=True)

with col2:
    clear_btn = st.button("🔄 Clear Input", use_container_width=True)

if clear_btn:
    st.rerun()

if send_btn and question:
    with st.spinner("🤔 Thinking..."):
        # Build context from previous chats
        context = ""
        if st.session_state.memory:
            context = "Previous conversations:\n"
            for chat in st.session_state.memory[-5:]:  # Use last 5 chats as context
                context += f"Q: {chat['question']}\nA: {chat['answer']}\n\n"
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant. Use the previous conversations as context if relevant.

{context}
Current question: {question}

Please provide a helpful and accurate response:"""
        
        # Get response
        answer = get_gemini_response(prompt)
        
        # Save to memory
        if not answer.startswith("Error"):
            new_chat = {
                "question": question,
                "answer": answer
            }
            st.session_state.memory.append(new_chat)
            save_chat_memory(st.session_state.memory)
        
        # Display results
        st.markdown("### 💡 Answer")
        st.write(answer)
        
        if context:
            st.markdown("---")
            st.markdown("### 📚 Context Used")
            st.text(context)

elif send_btn and not question:
    st.warning("Please enter a question first!")

# Display instructions if no activity
if not st.session_state.memory and not question:
    st.info("👆 Enter a question above to start chatting! Your conversations will be saved automatically.")