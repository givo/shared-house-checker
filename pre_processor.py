import os
import logging
from image_straightener import straighten_image
from pdf_to_image import load_pdf_pages


class PreProcessor:
    def __init__(self):
        pass

    def preprocess(self, pdf_path: str):
        logging.info("loading pdf file")
        pages = load_pdf_pages(pdf_path)

        for i in range(len(pages)):
            logging.info(f"processing image {i}")

            straight_image = straighten_image(pages[i])

            straight_image.show()
            input("Press Enter to continue...")
