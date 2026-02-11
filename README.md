# rag_chatbot

# Project Overview

This project is a Retrieval-Augmented Generation (RAG) prototype that assists UK bank compliance analysts in answering COREP regulatory reporting questions using official regulatory documents.

## The System

- Accepts a natural-language query
- Retrieves relevant PRA Rulebook / COREP instructions from a PDF
- Uses Gemini LLM to generate structured, schema-aligned output
- Produces audit-ready JSON suitable for regulatory reporting workflows

## 🎯 Problem Statement

UK banks subject to the PRA Rulebook must submit COREP regulatory returns that accurately reflect capital, risk exposures, and prudential metrics.

Preparing these returns is:

- Labour-intensive
- Error-prone
- Requires interpreting dense and frequently updated regulatory text

This prototype demonstrates how an LLM-assisted reporting assistant can:

- Retrieve relevant regulatory rules
- Populate structured COREP fields
- Provide traceable rule references and audit logs

## 🧠 Solution Approach (RAG Architecture)

The system follows an end-to-end RAG pipeline:

```
User Query
   ↓
Semantic Retrieval (ChromaDB)
   ↓
Relevant Regulatory Context
   ↓
Gemini LLM
   ↓
Structured JSON Output
```

## 🧩 Key Components

### 1️⃣ Document Ingestion

- Loads PRA / COREP regulatory PDF documents
- Each page is converted into structured text

### 2️⃣ Chunking

- Large documents are split into overlapping chunks
- Ensures context preservation and efficient retrieval

### 3️⃣ Embeddings

- Text chunks are converted into vector embeddings using `all-MiniLM-L6-v2`

### 4️⃣ Vector Database

- Embeddings are stored in ChromaDB
- Enables fast semantic search

### 5️⃣ Retrieval

- Top-K relevant chunks are retrieved for each query

### 6️⃣ LLM Generation

- Gemini (gemini-2.5-flash) generates responses
- Output strictly follows a predefined JSON schema
