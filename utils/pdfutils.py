import fitz  # PyMuPDF
from preprocessing.text_cleaning import detect_repeated_lines, remove_headers_footers

def is_scanned_pdf(pdf_path: str) -> bool:
    """Detect if PDF is scanned (no extractable text)."""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            if page.get_text("text").strip():
                return False
    return True

def extract_pdf_text(pdf_path: str):
    """Extract text from text-based PDFs with header/footer removal."""
    docs = []
    all_page_lines = []

    with fitz.open(pdf_path) as pdf:
        # Collect all page lines first
        for page in pdf:
            lines = page.get_text("text").split("\n")
            all_page_lines.append(lines)

        repeated_lines = detect_repeated_lines(all_page_lines)

        for page_num, page in enumerate(pdf):
            lines = page.get_text("text").split("\n")
            lines = remove_headers_footers(lines, repeated_lines)
            page_text = " ".join(lines).strip()

            if page_text:
                docs.append({
                    "content": page_text,
                    "metadata": {
                        "source": pdf_path,
                        "page": page_num + 1
                    }
                })
    return docs
