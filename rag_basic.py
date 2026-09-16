from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
import numpy as np


# =========================================================
# 1. EMBEDDING MODEL (SEARCH)
# =========================================================

embed_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embed_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def rewrite_query(query):
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small"
    )

    prompt = f"""
Rewrite the following question to be clear, specific,
and optimized for semantic search.

Original question:
{query}

Rewritten question:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=40
    )

    rewritten = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return rewritten.strip()


# =========================================================
# 2. CREATE EMBEDDING
# =========================================================

def get_embedding(text):
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = embed_model(**inputs)

    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding[0].numpy()


# =========================================================
# 3. LOAD DATA
# =========================================================

with open("data.txt", "r", encoding="utf-8") as f:
    chunks = [
        line.strip()
        for line in f
        if line.strip()
    ]

chunk_embeddings = [
    get_embedding(chunk)
    for chunk in chunks
]


# =========================================================
# 4. USER QUERY
# =========================================================

user_query = input("Ask a question: ")

query = user_query
print("Original:", user_query)


query_embedding = get_embedding(query)


# =========================================================
# 5. SIMILARITY SEARCH
# =========================================================

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


scores = [
    cosine_similarity(query_embedding, emb)
    for emb in chunk_embeddings
]


# Rank chunks by relevance
top_k = 3

top_indices = np.argsort(scores)[-top_k:][::-1]


print("\n--- RETRIEVED CHUNKS ---\n")

for i in top_indices:
    print(
        f"Score: {scores[i]:.3f} | {chunks[i]}"
    )


best_score = scores[top_indices[0]]


# =========================================================
# 6. CONFIDENCE GATE
# =========================================================

THRESHOLD = 0.45

if best_score < THRESHOLD:

    print("\n--- ANSWER ---\n")
    print(
        "I don't know. This information is not in my data."
    )

    exit()


# =========================================================
# 7. BUILD CONTEXT
# =========================================================

retrieved_context = chunks[top_indices[0]]

# =========================================================
# 8. GENERATION MODEL
# =========================================================

gen_tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

gen_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)


prompt = f"""
Use the context below to answer the question.

Give a short, direct answer based only on the context.
Do not use outside knowledge.
If the context does not contain the answer, say "I don't know."

Context:
{retrieved_context}

Question:
{query}

Answer:
"""


inputs = gen_tokenizer(
    prompt,
    return_tensors="pt"
)


output = gen_model.generate(
    **inputs,
    max_new_tokens=80
)


# =========================================================
# 9. FINAL ANSWER
# =========================================================

print("\n--- ANSWER ---\n")

print(
    gen_tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
)