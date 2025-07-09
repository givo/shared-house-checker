import os
from PIL import Image
from pdf2image import convert_from_path

Image.MAX_IMAGE_PIXELS = None

pdf_path = os.path.join(
    os.path.dirname(__file__), "data", "13755-BM-12847-31-[405]-G1.pdf"
)
pages = convert_from_path(pdf_path, dpi=300)  # or higher if needed
page = pages[0]  # Assuming first page is the one with 10 floors
page.save("full_page.jpg")
