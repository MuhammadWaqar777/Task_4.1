Context-Aware Chatbot Using LangChain or RAG
Overview

This project demonstrates the development of a context-aware conversational chatbot that can remember previous interactions and retrieve relevant information from external sources using LangChain or Retrieval-Augmented Generation (RAG).

Objective

To build a conversational chatbot capable of:

Remembering and utilizing conversational history (context memory)

Retrieving accurate answers from a vectorized document store

Integrating external knowledge sources (e.g., Wikipedia, internal documents, or any custom corpus)

Dataset

A custom corpus was used (e.g., Wikipedia pages, internal documents, or any chosen knowledge base).

Features

Context-aware conversation

Document embedding and vector search

Retrieval-Augmented Generation (RAG)

Streamlit-based web deployment

Instructions

Use LangChain or RAG for the chatbot framework.

Implement context memory to handle multi-turn conversations.

Retrieve answers from a vectorized document store.

Deploy the chatbot using Streamlit for interactive usage.

Skills Gained

Conversational AI development

Document embedding & vector search

Retrieval-Augmented Generation (RAG)

LLM integration & deployment

Deployment

The chatbot is deployed using Streamlit, providing a user-friendly web interface.

Tech Stack

Python

LangChain / RAG

Vector Store (e.g., FAISS, Pinecone, or Chroma)

Streamlit

How to Run

Clone the repository:

git clone https://github.com/your-username/context-aware-chatbot.git


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py
