import logging
from pathlib import Path

from app.utils.helpers import clean_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".html", ".htm"}


def parse_pdf(file_bytes: bytes, filename: str) -> str:
    try:
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        raw = "\n\n".join(pages)
        logger.info(f"Parsed PDF '{filename}': {len(reader.pages)} pages, {len(raw)} chars")
        return clean_text(raw)
    except Exception as e:
        logger.error(f"Failed to parse PDF '{filename}': {e}")
        return ""


def parse_text(file_bytes: bytes, filename: str) -> str:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")
    logger.info(f"Parsed text '{filename}': {len(text)} chars")
    return clean_text(text)


def parse_html(file_bytes: bytes, filename: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(file_bytes, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        logger.info(f"Parsed HTML '{filename}': {len(text)} chars")
        return clean_text(text)
    except Exception as e:
        logger.error(f"Failed to parse HTML '{filename}': {e}")
        return ""


def parse_document(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_bytes, filename)
    elif ext in (".html", ".htm"):
        return parse_html(file_bytes, filename)
    elif ext == ".txt":
        return parse_text(file_bytes, filename)
    else:
        logger.warning(f"Unsupported file type '{ext}' for '{filename}'")
        return ""


def get_page_count(file_bytes: bytes, filename: str) -> int | None:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            from io import BytesIO
            return len(PdfReader(BytesIO(file_bytes)).pages)
        except Exception:
            return None
    return None
