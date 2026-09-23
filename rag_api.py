import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_core_faiss import get_relevant_context
from rag_core_faiss import run_rag

load_dotenv()

API_KEY = os.getenv("API_KEY")

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def health():
    return {"status": "RAG API is running"}


@app.post("/ask")
def ask(req: QueryRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    answer = run_rag(req.question)
    return {"answer": answer}