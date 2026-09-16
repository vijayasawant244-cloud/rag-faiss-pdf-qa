from document_loader import load_pdf
from chunking import split_sentences, create_chunks

# Load PDF
text = load_pdf("documents/company.pdf")

# Split into sentences
sentences = split_sentences(text)

# Create chunks
chunks = create_chunks(
    sentences,
    sentences_per_chunk=3,
    overlap=1
)

print("Total sentences:", len(sentences))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk)