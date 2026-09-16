# RAG PDF Q&A System

A general-purpose Retrieval-Augmented Generation (RAG) application that answers questions from any PDF or text document — with built-in protection against hallucinated answers when information isn't actually present in the source document.

**Repo:** https://github.com/vijayasawant244-cloud/rag-faiss-pdf-qa

---

## What It Does

1. Extracts text from a PDF (or plain text file)
2. Cleans and splits the text into word-preserving chunks
3. Converts chunks into embeddings using a pretrained sentence embedding model
4. Stores embeddings in a FAISS vector index for fast similarity search
5. Retrieves the most relevant chunks for a user's question
6. Uses a pretrained language model to generate an answer grounded only in the retrieved context
7. Serves everything through a FastAPI endpoint

The system is domain-agnostic — it works with any PDF, not just one specific document, and contains no question-specific hardcoded logic.

---

## Architecture

PDF / Text File
↓
Text Extraction (PyPDF2)
↓
Text Cleaning
↓
Word-Based Chunking (no words cut mid-word)
↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2, mean pooling, normalized)
↓
FAISS Vector Index (IndexFlatIP — cosine similarity via normalized inner product)
↓
Similarity Search (top-k relevant chunks)
↓
Relevance Verification (two-layer check, see below)
↓
LLM Answer Generation (google/flan-t5-base)
↓
Final Answer (via FastAPI)


---

## Avoiding Hallucinated Answers

A key design goal of this project: if a question can't be answered from the document, the system should say so — not guess.

This is handled with a **two-layer, fully general check** (no hardcoded question or topic logic):

1. **Similarity threshold** — if the best-matching chunk's FAISS similarity score falls below a set threshold, the question is treated as unrelated to the document and rejected immediately, before any LLM call.
2. **LLM relevance check** — for questions that are topically close to the document but not actually answered by it (e.g. asking about a "favorite" language when the document only lists skills), a separate yes/no LLM call asks: *"Does this context explicitly contain the answer to this question?"* Only if the answer is yes does the system proceed to generate a final answer.

This two-step approach exists because similarity score alone isn't reliable — a topically related chunk can score highly even when it doesn't actually answer the specific question asked. Splitting "is this relevant?" from "what is the answer?" into two separate, simpler LLM tasks proved far more reliable than asking one model to judge relevance and generate an answer in a single pass.

If either check fails, the system returns:

I don't know based on the given data.


---

## Tech Stack

- **Python 3.14**
- **PyPDF2** — PDF text extraction
- **Hugging Face Transformers** — tokenization and model inference
- **sentence-transformers/all-MiniLM-L6-v2** — embedding model (used via raw `transformers` API with manual mean pooling, not the `sentence-transformers` library wrapper)
- **FAISS (faiss-cpu)** — vector similarity search (`IndexFlatIP`)
- **google/flan-t5-base** — pretrained sequence-to-sequence LLM for answer generation
- **FastAPI** — REST API
- **Uvicorn** — ASGI server
- **Pydantic** — request validation

No model is trained from scratch — all models are pretrained and used purely for inference.

---

## Project Structure

rag-fastapi/
├── chunking.py # PDF/text extraction, cleaning, word-based chunking
├── document_loader.py # Document loading utilities
├── embeddings_basic.py # Embedding experiments
├── faiss_store.py # FAISS index utilities
├── llm_basic.py # LLM experiments
├── local_llm.py # Local LLM utilities
├── rag_api.py # FastAPI app and /ask endpoint
├── rag_basic.py # Early/basic RAG implementation
├── rag_core_faiss.py # Main RAG pipeline (embedding, FAISS, generation, relevance checks)
├── rag_core_naive.py # Naive baseline RAG implementation
├── tests/ # Test files
├── requirements.txt
└── README.md


---

## Setup

```powershell
# Clone the repo
git clone https://github.com/vijayasawant244-cloud/rag-faiss-pdf-qa.git
cd rag-faiss-pdf-qa

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Place your PDF in the `documents/` folder (this folder is git-ignored, so add your own file locally).

---

## Running the API

```powershell
uvicorn rag_api:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs are auto-generated at `http://127.0.0.1:8000/docs`.

---

## Example Usage

**Request:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"question": "What programming languages does the person know?"}'
```

**Response:**
```json
{
  "answer": "Python, SQL, Machine Learning, Computer Vision, and NLP"
}
```

**Unanswerable question:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"question": "What is the capital of France?"}'
```

**Response:**
```json
{
  "answer": "I don't know based on the given data."
}
```

---

## Key Learnings

- Word-based chunking prevents mid-word truncation compared to naive character-based chunking
- Normalized embeddings + `IndexFlatIP` in FAISS approximate cosine similarity efficiently
- High embedding similarity does not guarantee an answer is actually present — topic overlap and factual relevance are different things
- Small-to-medium instruction-following models (e.g. FLAN-T5) are more reliable at answering a simple yes/no relevance question than at judging their own uncertainty while generating an open-ended answer in the same pass

