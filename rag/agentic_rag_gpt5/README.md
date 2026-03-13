# 🧠 Agentic RAG with GPT-5

This is a simple and easy-to-use RAG (Retrieval-Augmented Generation) application made with the Agno framework. It uses GPT-5 from OpenAI and LanceDB to help you quickly find information and get clear answers to your questions.

## ✨ Features

- **🤖 GPT-5**: Uses the latest OpenAI model to give smart and relevant answers.
- **🗄️ LanceDB**: Stores and searches documents quickly using vectors.
- **🔍 Agentic RAG**: Combines document search with AI to improve answers.
- **📝 Markdown Formatting**: Answers are nicely formatted and easy to read.
- **🌐 Add Your Own Documents**: Quickly add web links to expand the knowledge base.
- **⚡ Live Answer Streaming**: See your answer build up on the screen as it's generated.
- **🎯 Simple Interface**: Just focus on asking and getting answers—no complex setup.

## 🚀 Quick Start

### What You Need

- Python 3.11 or newer
- An OpenAI API key that can use GPT-5

## 🎯 How to Use

1. **Enter your OpenAI API key** in the sidebar.
2. **Add links to knowledge sources** using the sidebar (just paste in URLs).
3. **Type your question** or choose a suggested prompt.
4. **See your answer appear in real time**, nicely formatted.

### Example Questions

- **"What is Agno?"** – Find out what the Agno framework and its agents are.
- **"Teams in Agno"** – Learn how teams work within the Agno system.
- **"Build RAG system"** – Get simple steps to create your own RAG (Retrieval-Augmented Generation) system.

## 🏗️ How It Works

### Main Parts

- **`Agent`**: Handles asking and answering questions.
- **`UrlKnowledge`**: Loads documents from web links.
- **`LanceDb`**: Stores and searches documents by meaning.
- **`OpenAIEmbedder`**: Turns text into vectors for searching.
- **`OpenAIChat`**: Uses the GPT-5-nano model to write answers.

### Process Overview

1. **Load Knowledge**: Links you provide are read and added to LanceDB.
2. **Semantic Search**: Your question is turned into a vector and compared to find matching documents.
3. **Answer Generation**: GPT-5-nano reads the best matches and writes a helpful answer.
4. **Live Output**: The answer appears on your screen as it’s created.

## 🔧 Settings

### Database

- **Type**: LanceDB (files stored on your computer)
- **Table Name**: `agentic_rag_docs`
- **Search**: Finds best document matches using vector similarity

## 📚 Managing Knowledge

### Add More Sources

- Use the sidebar to paste in new URLs.
- Each source is added and indexed automatically.
- All your sources are shown in a numbered list.

### Initial Knowledge

- By default, starts with Agno's main documentation: `https://docs.agno.com/introduction/agents.md`
- You can add any web documentation you like.

## 🎨 User Interface

### Sidebar

- **API Key**: Secure way to add your OpenAI API key.
- **Add URLs**: Easily add more web links for knowledge.
- **View Sources**: See all loaded URLs as a list.

### Main Area

- **Suggested Prompts**: Click to quickly ask common questions.
- **Question Input**: Type any question in a big text area.
- **Live Answers**: Watch the answer appear as the AI writes it.
- **Formatted Responses**: Answers use markdown for clear formatting.
