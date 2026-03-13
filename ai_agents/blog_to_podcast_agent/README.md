## 📰 ➡️ 🎙️ Blog to Podcast Agent

This Streamlit app makes it easy to turn any blog post into a podcast episode. Simply enter the URL of a public blog, and the app will automatically scrape the content (using Firecrawl), summarize it with OpenAI's GPT-4, and generate an audio version using the ElevenLabs API.

## Features

- **Blog Scraping**: Automatically extracts the full content from any public blog URL with the Firecrawl API.
- **Summary Generation**: Produces a concise and engaging summary (up to 2000 characters) using OpenAI GPT-4.
- **Podcast Creation**: Converts your summary into a natural-sounding podcast with the ElevenLabs voice API.
- **Secure API Key Input**: Easily add your OpenAI, Firecrawl, and ElevenLabs API keys securely via the sidebar.

## Setup

### Requirements

1. **API Keys**:
    - **OpenAI API Key**: Sign up at OpenAI to get your API key.
    - **ElevenLabs API Key**: Register at ElevenLabs for your API key.
    - **Firecrawl API Key**: Apply for access and get your API key from Firecrawl.

2. **Python 3.8 or Newer**: Make sure your system is running Python version 3.8 or above.
