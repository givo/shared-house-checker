import cv2
from PIL import Image
import numpy as np


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def show_image_cv(img_pil: Image.Image):
    img_cv = pil_to_cv2(img_pil)
    cv2.imshow("Image", img_cv)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
