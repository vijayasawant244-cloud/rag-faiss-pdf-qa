from pathlib import Path
import re
import PyPDF2


def extract_pdf_text(file_path: str) -> str:
    text = ""

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")

    # Fix spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Keep meaningful line breaks
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 450,
    overlap: int = 80
):
    """
    Split text by complete words.
    Words will not be cut in the middle.
    """

    words = text.split()

    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:

        word_length = len(word) + 1

        if (
            current_length + word_length > chunk_size
            and current_chunk
        ):
            chunks.append(" ".join(current_chunk))

            # Keep some previous words for overlap
            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_chunk):
                overlap_length += len(previous_word) + 1
                overlap_words.insert(0, previous_word)

                if overlap_length >= overlap:
                    break

            current_chunk = overlap_words
            current_length = sum(
                len(item) + 1
                for item in current_chunk
            )

        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def load_and_chunk_file(
    file_path: str,
    chunk_size: int = 450,
    overlap: int = 80
):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() == ".pdf":
        text = extract_pdf_text(file_path)
    else:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    text = clean_text(text)

    return split_into_chunks(
        text,
        chunk_size=chunk_size,
        overlap=overlap
    )