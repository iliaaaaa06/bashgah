"""Extract text from PDF / TXT / DOCX files containing Persian text."""
import io
from dataclasses import dataclass

import pymupdf
from docx import Document as DocxDocument

from app.persian import normalize

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# cp1256 (Windows-Arabic) is still common for older Persian .txt files
_TXT_ENCODINGS = ("utf-8-sig", "cp1256")


class DocumentReadError(Exception):
    pass


@dataclass
class Page:
    text: str
    page: int | None  # 1-based page number for PDFs, None otherwise


def load_document(data: bytes, extension: str) -> list[Page]:
    extension = extension.lower()
    try:
        if extension == ".pdf":
            pages = _load_pdf(data)
        elif extension == ".docx":
            pages = _load_docx(data)
        elif extension == ".txt":
            pages = [Page(_decode_txt(data), None)]
        else:
            raise DocumentReadError(extension)
    except DocumentReadError:
        raise
    except Exception as exc:  # corrupt / encrypted / malformed file
        raise DocumentReadError(str(exc)) from exc

    normalized = [Page(normalize(p.text), p.page) for p in pages]
    return [p for p in normalized if p.text]


def _load_pdf(data: bytes) -> list[Page]:
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            # sort=True restores reading order for multi-column / RTL layouts
            pages.append(Page(page.get_text("text", sort=True), i))
    return pages


def _load_docx(data: bytes) -> list[Page]:
    doc = DocxDocument(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = []
            for cell in row.cells:
                text = cell.text.strip()
                if text and (not cells or cells[-1] != text):  # merged cells repeat their text
                    cells.append(text)
            if cells:
                parts.append(" | ".join(cells))
    return [Page("\n".join(parts), None)]


def _decode_txt(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in _TXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
