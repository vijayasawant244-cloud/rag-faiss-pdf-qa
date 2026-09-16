from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

context = """
Deep learning uses neural networks with many layers.
"""

question = "What is deep learning?"

prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=50
)

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Answer:", answer)