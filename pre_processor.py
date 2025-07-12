import os
from image_straightener import straighten_image
from pdf_to_image import load_pdf_pages


class PreProcessor:
    def __init__(self):
        os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"

    def preprocess(self, pdf_path: str):
        images = load_pdf_pages(pdf_path)
        straight_image = straighten_image(images[0])

        straight_image.show()
        input("Press Enter to continue...")


if __name__ == "__main__":
    pre_processor = PreProcessor()

    # image_path = os.path.join(os.path.dirname(__file__), "data", "floor1.jpg")
    image_path = os.path.join(
        os.path.dirname(__file__), "data", "sketches", "or_akiva_mor_11.pdf"
    )

    pre_processor.preprocess(image_path)
