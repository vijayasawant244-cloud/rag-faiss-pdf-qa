import faiss
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM
)

from chunking import load_and_chunk_file


# =========================================================
# 1. EMBEDDING MODEL - MiniLM
# =========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embed_tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL
)


def get_embedding(text: str):
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = embed_model(**inputs)

    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding[0].numpy().astype("float32")


# =========================================================
# 2. LOAD PDF + CHUNKS
# =========================================================

CHUNKS = load_and_chunk_file(
    "documents/company.pdf"
)

print("Total chunks:", len(CHUNKS))


# =========================================================
# 3. CREATE EMBEDDINGS
# =========================================================

embeddings = np.vstack([
    get_embedding(chunk)
    for chunk in CHUNKS
])


# =========================================================
# 4. NORMALIZE EMBEDDINGS
# =========================================================

embeddings = embeddings / np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)


# =========================================================
# 5. FAISS VECTOR SEARCH
# =========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# =========================================================
# 6. LLM - QWEN 0.5B
# =========================================================

LLM_MODEL = "google/flan-t5-base"

gen_tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

gen_model = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL
)


# =========================================================
# 7. RAG FUNCTION
# =========================================================

def is_context_relevant(context: str, user_query: str) -> bool:
    """
    Ask the LLM a simple yes/no question:
    does this context actually contain the answer?
    This is a separate, easier task than generating the answer itself.
    """

    check_prompt = f"""Context:
{context}

Question: {user_query}

Does the context above explicitly contain the answer to this question? Answer with only one word: Yes or No.

Answer:"""

    inputs = gen_tokenizer(
        check_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    verdict = gen_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip().lower()

    print("Relevance check verdict:", verdict)

    return verdict.startswith("yes")


def run_rag(user_query: str):
    
    # -----------------------------------------------------
    # Query → embedding
    # -----------------------------------------------------

    query_emb = get_embedding(
        user_query
    ).reshape(1, -1)

    query_emb = query_emb / np.linalg.norm(
        query_emb,
        axis=1,
        keepdims=True
    )

    # -----------------------------------------------------
    # Search relevant chunks
    # -----------------------------------------------------

    D, I = index.search(
        query_emb,
        k=min(3, len(CHUNKS))
    )

    similarity_score = float(D[0][0])

    print("Similarity:", similarity_score)

    # -----------------------------------------------------
    # Relevance threshold (cheap first filter)
    # -----------------------------------------------------

    if similarity_score < 0.20:
        return "I don't know based on the given data."

    # -----------------------------------------------------
    # Retrieve context
    # -----------------------------------------------------

    context = "\n".join(
        f"- {CHUNKS[i]}"
        for i in I[0]
    )

    print("Retrieved context:")
    print(context)

    # -----------------------------------------------------
    # Relevance check (second, stronger filter)
    # -----------------------------------------------------

    if not is_context_relevant(context, user_query):
        return "I don't know based on the given data."

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the question using ONLY the provided context.

Rules:
- Use only facts explicitly present in the context.
- Do not use outside knowledge.
- Do not guess or invent information.
- Answer the exact question asked.
- Do not add unrelated details.
- Give a complete but concise answer.
- If the answer is not present in the context, say exactly:
I don't know based on the given data.

Context:
{context}

Question:
{user_query}

Answer:
"""
    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.no_grad():
        outputs = gen_model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

    # Decode only newly generated tokens
    answer = gen_tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
    ).strip()

    if not answer:
        return "I don't know based on the given data."

    return answer


# =========================================================
# 8. GET RELEVANT CONTEXT
# =========================================================

def get_relevant_context(user_query: str):

    query_emb = get_embedding(
        user_query
    ).reshape(1, -1)

    query_emb = query_emb / np.linalg.norm(
        query_emb,
        axis=1,
        keepdims=True
    )

    D, I = index.search(
        query_emb,
        k=min(3, len(CHUNKS))
    )

    score = float(D[0][0])

    if score < 0.20:
        return "I don't know based on the given data.", score

    context = "\n".join(
        CHUNKS[i]
        for i in I[0]
    )

    return context, score