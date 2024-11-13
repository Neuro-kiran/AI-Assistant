# AI Assistant with Enhanced RAG (Retrieval-Augmented Generation)

This project demonstrates the creation of an AI assistant that leverages Retrieval-Augmented Generation (RAG) for answering user questions. The assistant utilizes large language models and document processing techniques to answer questions based on the context of uploaded documents (PDF and TXT).

## Features

- **Document Processing**: Upload PDF or TXT files and process them into manageable chunks for better question answering.
- **Retrieval-Augmented Generation**: Use the enhanced RAG technique to retrieve relevant document chunks and generate responses based on them.
- **AI-Powered Responses**: The assistant answers general questions and provides document-based responses if relevant context is available.
- **User-Friendly Interface**: Built using Streamlit for easy and interactive user experience.

## Requirements

To run this project locally, ensure you have the following installed:

- Python 3.7 or higher
- Streamlit
- LangChain
- FAISS
- Hugging Face
- Groq API (for using ChatGroq)
  
You can install the required dependencies by running:

```bash
pip install streamlit langchain langchain-community faiss-cpu huggingface_hub
