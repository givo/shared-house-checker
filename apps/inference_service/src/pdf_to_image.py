import os
from typing import List
from PIL import Image
from pdf2image import convert_from_path

Image.MAX_IMAGE_PIXELS = None


def pdf_page_to_image(pdf_filename, output_image_path, page_number=0, dpi=300):
    """
    Converts a page from a PDF file to an image and saves it.

    Args:
        pdf_filename (str): Path to the PDF file.
        output_image_path (str): Path to save the output image.
        page_number (int): Page number to convert (0-based index).
        dpi (int): DPI for conversion.
    """
    pages = convert_from_path(pdf_filename, dpi=dpi)
    page = pages[page_number]
    page.save(output_image_path)


def load_pdf_pages(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Converts a multi-page PDF to a list of PIL Images (in memory).
    Each page becomes one image.
    """
    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Invalid PDF path")
    return convert_from_path(pdf_path, dpi=dpi)
