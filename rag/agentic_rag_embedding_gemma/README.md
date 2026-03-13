## 🔥 Agentic RAG with EmbeddingGemma

This Streamlit app is a simple example of an Agentic Retrieval-Augmented Generation (RAG) Agent. It uses Google's EmbeddingGemma model to create vector embeddings and Llama 3.2 as the language model. Both models run locally using Ollama.

### Features

- **Works Locally**: Uses EmbeddingGemma for searching by meaning and Llama 3.2 for generating answers, all on your own computer.
- **PDF Database**: You can add links to PDF files to build up a knowledge base.
- **Fast Search**: Uses LanceDB to quickly find documents similar to your questions.
- **Easy-to-Use Interface**: Friendly Streamlit UI lets you add PDFs and ask questions.
- **Live Responses**: See responses generated in real time with tool usage shown.

### How Does It Work?

1. **Add PDFs**: Enter PDF URLs in the sidebar to load and index their contents.
2. **Generate Embeddings**: EmbeddingGemma turns the text in your PDFs into vectors for advanced search.
3. **Ask Questions**: Your question is also turned into a vector and compared with the database.
4. **Get Answers**: Llama 3.2 reads the relevant information and writes an answer.
5. **See the Tools**: The agent uses search tools to find the best information before answering.

### Requirements

- Python 3.8 or newer
- Ollama installed and running
- Download the models: `embeddinggemma:latest`, `llama3.2:latest`

### Technologies Used

- **Agno**: Framework for building agents
- **Streamlit**: For the web interface
- **LanceDB**: For vector storage and search
- **Ollama**: Runs models locally
- **EmbeddingGemma**: Embedding model from Google
- **Llama 3.2**: Language model from Meta
