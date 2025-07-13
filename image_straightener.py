import re
import os
import logging
from typing import Tuple, List
import numpy as np
import cv2
from PIL import Image
import pytesseract

Image.MAX_IMAGE_PIXELS = None
os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"


class Desker:
    HEBREW_CHAR_RE = re.compile(r"[\u0590-\u05FF]")

    def pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def extract_ocr_region(self, image: np.ndarray, max_dim: int = 2048) -> np.ndarray:
        h, w = image.shape[:2]
        aspect_ratio = abs(h / w)
        if h > w:
            crop_h = int(h // aspect_ratio)
            y_start = int((h - crop_h) // 2)
            crop = image[y_start : y_start + crop_h, :]
        else:
            crop_w = int(w // aspect_ratio)
            x_start = int((w - crop_w) // 2)
            crop = image[:, x_start : x_start + crop_w]
        ch, cw = crop.shape[:2]
        scale = min(max_dim / ch, max_dim / cw, 1.0)
        if scale < 1.0:
            new_size = (int(cw * scale), int(ch * scale))
            crop = cv2.resize(crop, new_size, interpolation=cv2.INTER_AREA)
        return crop

    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return image
        rotation_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[angle]
        return cv2.rotate(image, rotation_code)

    def is_valid_hebrew_word(self, text: str) -> bool:
        text = text.strip()
        if len(text) < 2:
            return False
        hebrew_letters = self.HEBREW_CHAR_RE.findall(text)
        return len(hebrew_letters) >= 2

    def score_rotation(self, crop: np.ndarray) -> float:
        data = pytesseract.image_to_data(
            crop, lang="heb+eng", output_type=pytesseract.Output.DICT
        )
        confidences = [
            int(conf)
            for text, conf in zip(data["text"], data["conf"])
            if int(conf) > 0 and self.is_valid_hebrew_word(text)
        ]
        if not confidences:
            return 0.0
        avg_conf = np.mean(confidences)
        word_bonus = np.log(len(confidences) + 1)
        return float(avg_conf * word_bonus)

    def auto_orient_image(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        ocr_crop = self.extract_ocr_region(image)
        best_angle = 0
        best_score = 0.0
        for angle in [0, 90, 180, 270]:
            rotated_crop = self.rotate_image(ocr_crop, angle)
            logging.info(f"Scoring rotation {angle} degrees")
            # cv2.imshow("OCR Crop", rotated_crop)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()
            score = self.score_rotation(rotated_crop)
            if score > best_score:
                best_score = score
                best_angle = angle
        return self.rotate_image(image, best_angle), best_angle

    def deskew(self, pil_image: Image.Image) -> Image.Image:
        cv_image = self.pil_to_cv2(pil_image)
        corrected_cv_image, angle = self.auto_orient_image(cv_image)
        logging.info(f"Rotated by {angle} degrees")
        return self.cv2_to_pil(corrected_cv_image)
