import logging
import os
from pre_processor import PreProcessor


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


if __name__ == "__main__":
    pre_processor = PreProcessor()

    # image_path = os.path.join(os.path.dirname(__file__), "data", "floor1.jpg")
    image_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "sketches",
        "herzelia_hachalil_2_3_4_5_6_7_8.pdf",
    )

    pre_processor.preprocess(image_path)
