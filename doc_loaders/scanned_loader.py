# doc_loaders/scanned_loader.py
import fitz
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from preprocessing.image_cleaning import clean_image
from preprocessing.text_cleaning import postprocess_text, extract_ayurveda_entities
import re 

def ocr_extract_english(image):
    """
    Run OCR with bounding boxes and keep only English-like words.
    This removes Sanskrit and gibberish while preserving valid English text.
    """
    data = pytesseract.image_to_data(image, lang="eng+san", output_type=Output.DICT)

    english_words = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue

        # Count English-like characters
        english_chars = sum(c.isalnum() or c in ".,;:()'\"-" for c in word)
        ratio = english_chars / max(1, len(word))

        if (
            re.fullmatch(r"[A-Za-z0-9.,;:()'\"-]+", word)  # valid English-looking token
            and ratio >= 0.8                              # mostly English chars
            and len(word) <= 25                           # avoid very long OCR junk
        ):
            english_words.append(word)

    return " ".join(english_words)


def load_scanned_pdf(pdf_path):
    """
    Extract text from scanned PDFs page by page.
    Uses image preprocessing + OCR filtering to return only English text.
    """
    docs = []
    pdf_document = fitz.open(pdf_path)

    for page_num, page in enumerate(pdf_document):
        # Convert page to image
        pix = page.get_pixmap(dpi=200, alpha=False)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:  # Handle RGBA
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)

        # Preprocess image for OCR
        img_cleaned = clean_image(img_data)

        # Run OCR (English filter applied)
        text = ocr_extract_english(img_cleaned)

        # Postprocess with same cleaning pipeline as textual PDFs
        cleaned_text = postprocess_text(
            text, page_num=page_num + 1, pdf_name=pdf_path
        )

        # Ensure no empty outputs
        if not cleaned_text.strip():
            cleaned_text = text

        entities = extract_ayurveda_entities(cleaned_text)

        docs.append({
            "content": cleaned_text,
            "metadata": {
                "source": pdf_path,
                "page": page_num + 1,
                "entities": entities
            }
        })

    pdf_document.close()
    return docs
