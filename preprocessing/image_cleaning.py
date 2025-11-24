import cv2
import numpy as np

def clean_image(image):
    """
    Preprocess scanned image for better OCR results.
    Steps:
    1. Convert to grayscale
    2. Denoise (median blur + non-local means)
    3. Adaptive thresholding
    4. Morphological opening/closing to remove small noise
    5. Resize (optional) for sharper OCR
    """
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.fastNlMeansDenoising(gray, None, h=30)

    # 3. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # 4. Morphological cleaning
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. Upscale if small
    h, w = clean.shape
    if h < 1000:
        clean = cv2.resize(clean, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    return clean
