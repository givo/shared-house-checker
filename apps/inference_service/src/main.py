import logging
import os
import cv2
from detect_apt_centroid import detect_apartment_circles
from image_utils import pil_to_cv2
from pre_processor import PreProcessor


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


if __name__ == "__main__":
    pre_processor = PreProcessor()

    image_path = os.path.join(os.path.dirname(__file__), "data", "part_of_floor_0.png")
    logging.info(f"Processing image: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    detect_apartment_circles(image, True)

    # image_path = os.path.join(
    #     os.path.dirname(__file__),
    #     "data",
    #     "sketches",
    #     "herzelia_hachalil_2_3_4_5_6_7_8.pdf",
    # )
    # sketches_dir = os.path.join(os.path.dirname(__file__), "data", "sketches")
    # for filename in os.listdir(sketches_dir):
    #     file_path = os.path.join(sketches_dir, filename)
    #     if os.path.isfile(file_path):
    #         logging.info(f"Processing image: {os.path.basename(file_path)}")
    #         pre_processor.preprocess(file_path)
