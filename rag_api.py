import os
import logging
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_core_faiss import get_relevant_context
from rag_core_faiss import run_rag

load_dotenv()

API_KEY = os.getenv("API_KEY")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong. Please try again later."}
    )


@app.get("/")
def health():
    return {"status": "RAG API is running"}


@app.post("/ask")
def ask(req: QueryRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    answer = run_rag(req.question)
    return {"answer": answer}