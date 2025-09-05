import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract_text

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    First tries PyPDF2, falls back to pdfminer if necessary.
    """
    text = ""

    # Try PyPDF2
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception:
        text = ""

    # Fallback to pdfminer if PyPDF2 fails
    if not text.strip():
        try:
            text = pdfminer_extract_text(file_path)
        except Exception:
            text = ""

    return text.strip()
