import pytesseract
from PIL import Image

class OcrAgent:
    def __init__(self, tesseract_cmd="tesseract"):
        # Set the Tesseract command path
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image_path):
        """
        Extracts text from an image using Tesseract OCR.
        Args:
            image_path (str): Path to the image file.
        Returns:
            str: Extracted text from the image.
        """
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            return f"Error during OCR: {str(e)}"