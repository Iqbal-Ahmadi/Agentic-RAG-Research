import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .pdf_ingest import extract_pdf_pages, chunk_text

@dataclass
class ChunkMeta:
    source: str
    page: int
    text: str

class VectorIndex:
    """
    Single responsibility:
      - Build/search a FAISS cosine-similarity index over chunks.
    """
    def __init__(self, embed_model_name: str):
        self.embedder = SentenceTransformer(embed_model_name)
        self.index = None  # faiss.Index
        self.meta: List[ChunkMeta] = []

    def build_from_pdf_dir(self, pdf_dir: str, chunk_chars: int, chunk_overlap: int) -> None:
        if not os.path.isdir(pdf_dir):
            raise ValueError(f"PDF directory not found: {pdf_dir}")

        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            raise ValueError("No PDF files found in the PDF directory.")

        texts: List[str] = []
        meta: List[ChunkMeta] = []

        for pdf_file in pdf_files:
            path = os.path.join(pdf_dir, pdf_file)
            pages = extract_pdf_pages(path)

            # Skipping scanned/protected PDFs (no extractable text)
            if not pages:
                print(f"WARNING: No extractable text in {pdf_file}. Skipping (scanned/protected).")
                continue

            for page_num, page_text in pages:
                for chunk in chunk_text(page_text, chunk_chars, chunk_overlap):
                    texts.append(chunk)
                    meta.append(ChunkMeta(source=pdf_file, page=page_num, text=chunk))

        if not texts:
            raise ValueError("No chunks extracted. Use text-based PDFs (not scanned images).")

        X = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
        faiss.normalize_L2(X)

        dim = X.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(X)
        self.meta = meta

    def retrieve(self, query: str, top_k: int) -> Tuple[str, List[Dict]]:
        if self.index is None or not self.meta:
            raise RuntimeError("Index is not built. Call build_from_pdf_dir() first.")

        qvec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qvec)

        scores, ids = self.index.search(qvec, top_k)
        ids = ids[0].tolist()

        retrieved = []
        for i in ids:
            if i == -1:
                continue
            m = self.meta[i]
            retrieved.append({"source": m.source, "page": m.page, "text": m.text})

        context = self._format_context(retrieved)
        return context, retrieved

    @staticmethod
    def _format_context(retrieved: List[Dict]) -> str:
        blocks = []
        for d in retrieved:
            blocks.append(f"[{d['source']} p.{d['page']}]\n{d['text']}")
        return "\n\n".join(blocks)
