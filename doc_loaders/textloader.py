# doc_loaders/textloader.py
import fitz
from preprocessing.text_cleaning import postprocess_text, extract_ayurveda_entities

def load_text_pdf(pdf_path):
    docs = []
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf):
            raw_text = page.get_text("text") or ""
            cleaned_text = postprocess_text(raw_text)
            if not cleaned_text.strip():
                cleaned_text = raw_text  # fallback if cleaning removes everything

            entities = extract_ayurveda_entities(cleaned_text)

            docs.append({
                "content": cleaned_text,
                "metadata": {
                    "source": pdf_path,
                    "page": page_num + 1,
                    "entities": entities
                }
            })
    return docs
