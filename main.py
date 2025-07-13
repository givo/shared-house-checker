import logging
import os
from pre_processor import PreProcessor


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


if __name__ == "__main__":
    logging.info("This is a test log message.")

    pre_processor = PreProcessor()

    # image_path = os.path.join(os.path.dirname(__file__), "data", "floor1.jpg")
    image_path = os.path.join(
        os.path.dirname(__file__), "data", "sketches", "raanana_shvil_hazahav_6.pdf"
    )

    pre_processor.preprocess(image_path)
