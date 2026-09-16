from pypdf import PdfReader


def load_pdf(filename):
    reader = PdfReader(filename)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text