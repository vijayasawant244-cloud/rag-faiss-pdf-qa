from fastapi import FastAPI
from pydantic import BaseModel
from rag_core_faiss import get_relevant_context
from rag_core_faiss import run_rag

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "RAG API is running"}


@app.post("/ask")
def ask(req: QueryRequest):
    answer = run_rag(req.question)
    return {"answer": answer}