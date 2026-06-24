"""
Extracts plain text from PDF and DOCX files.
Falls back to OCR (pytesseract) if a PDF has no selectable text (scanned pages).
"""

import io
from pathlib import Path

import PyPDF2
from docx import Document


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return _extract_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or DOCX.")


def _extract_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    full_text = "\n".join(pages).strip()

    # If no selectable text, try OCR
    if len(full_text) < 100:
        full_text = _ocr_pdf(file_bytes)

    return full_text


def _ocr_pdf(file_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
        import fitz  # PyMuPDF - optional fallback

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(pytesseract.image_to_string(img))
        return "\n".join(pages)
    except ImportError:
        return "[OCR unavailable — install pytesseract and PyMuPDF for scanned PDF support]"


def _extract_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
