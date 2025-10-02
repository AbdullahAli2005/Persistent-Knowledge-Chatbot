# 🤖 Persistent Knowledge Base Chatbot

A Streamlit-based chatbot powered by Google's Gemini AI that maintains persistent memory of conversations using FAISS vector storage. The chatbot remembers previous interactions and uses them to provide contextually relevant answers.

## 🌟 Features

* **Gemini AI Integration**: Powered by Google's Gemini 1.5 Flash model for intelligent responses
* **Persistent Memory**: Stores conversation history in a vector database (FAISS)
* **Semantic Search**: Finds relevant past conversations using embeddings
* **Streamlit UI**: Clean, interactive web interface
* **Memory Management**: View and clear chat history
* **Context-Aware Responses**: Uses relevant past Q/A pairs to inform current answers

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* Google API Key for Gemini AI

### Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd persistent-knowledge-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables:

* Create a `.env` file in the root directory
* Add your Google API key:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at [http://localhost:8501](http://localhost:8501).

## 📁 Project Structure

```text
persistent-knowledge-chatbot/
├── app.py                 # Main Streamlit application
├── memory_manager.py      # Memory management and FAISS operations
├── data/                  # Chat memory storage directory
│   └── chat_memory.json   # Persistent chat history
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
GOOGLE_API_KEY=your_actual_google_api_key
```

### Getting a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com)
2. Create a new API key
3. Copy the key to your `.env` file

## 💾 How It Works

### Memory System

* **Vector Embeddings**: Uses Gemini's embedding model to convert Q/A pairs into vectors
* **FAISS Index**: Efficient similarity search for retrieving relevant context
* **JSON Storage**: Persistent storage of chat history with embeddings
* **Semantic Search**: Finds the most relevant past conversations based on semantic similarity

### Chat Process

1. User asks a question
2. System searches for relevant past Q/A pairs using semantic similarity
3. Gemini AI generates answer using context from relevant past conversations
4. New Q/A pair is stored in memory with embeddings
5. Memory is persisted for future sessions

## 🛠️ Technical Details

### Components

* **GeminiLLM**: Wrapper class for Google's Gemini AI
* **MemoryManager**: Handles vector storage, retrieval, and persistence
* **FAISS**: Facebook AI Similarity Search for efficient vector operations
* **Streamlit**: Web framework for the user interface

### Models Used

* **Primary LLM**: gemini-1.5-flash for chat responses
* **Embedding Model**: models/embedding-001 for vector embeddings

## 📊 Usage

### Asking Questions

* Type your question in the text area
* Click **Send** to get an AI-generated response
* The chatbot will use relevant past conversations to inform its answer

### Viewing Memory

* The sidebar shows all previous conversations
* Recent chats appear at the top
* Context used for current answer is displayed below the response

### Managing Memory

* **Clear Chat**: Use the "Clear chat memory" button to delete all history
* **Persistent Storage**: Memory is automatically saved and reloaded between sessions

## 🔒 Privacy & Data

* All chat data is stored locally in the `data/` directory
* No data is sent to external services except Google's Gemini API
* You can clear all data at any time using the clear function

## 🐛 Troubleshooting

### Common Issues

* **API Key Error**: Ensure `GOOGLE_API_KEY` is set in `.env`
* **Module Not Found**: Run `pip install -r requirements.txt`
* **Memory Not Persisting**: Check write permissions in the `data/` directory
* **Embedding Errors**: Verify internet connection and API quota

### Logs

* Check Streamlit logs in the terminal for detailed error messages
* For Streamlit Cloud, view logs through the "Manage app" interface

## 🌐 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repo to Streamlit Cloud
3. Add `GOOGLE_API_KEY` as a secret in your app settings
4. Deploy!

### Other Platforms

The app can be deployed on any platform that supports Python and Streamlit. Ensure:

* Environment variables are properly set
* Write permissions for the `data/` directory
* Internet access for Gemini API calls

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

If you encounter any problems or have questions:

* Check the troubleshooting section above
* Open an issue on GitHub
* Ensure your dependencies are up to date
