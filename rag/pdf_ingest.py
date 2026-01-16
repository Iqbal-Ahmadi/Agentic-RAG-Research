import re
import pdfplumber
from typing import List, Tuple

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if len(chunk) >= 80:  # ignore tiny fragments
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number starting from 1, cleaned_text)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            txt = clean_text(txt)
            if txt:
                pages.append((i, txt))
    return pages
