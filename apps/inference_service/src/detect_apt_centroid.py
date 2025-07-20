import cv2
import numpy as np
import pytesseract
import re


def detect_apartment_circles(image: np.ndarray, debug=False):
    config = '--psm 7 -c tessedit_char_whitelist=0123456789אבגדהוזחטיכלמנסעפצקרשת"׳'
    print(config)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # HoughCircles params — tweak as needed
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=60,
        minRadius=3,
        maxRadius=120,
    )

    results = []
    if circles is not None and len(circles) > 0:
        circles = np.around(circles[0]).astype(np.uint16)  # shape (N, 3) or (1, 3)
        if circles.ndim == 1 and circles.shape[0] == 3:
            circles = [circles]  # wrap single row in list

        for x, y, r in circles:

            # Extract circle ROI
            margin = int(r * 1.1)
            x1, x2 = max(x - margin, 0), min(x + margin, image.shape[1])
            y1, y2 = max(y - margin, 0), min(y + margin, image.shape[0])
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # OCR inside ROI
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(
                roi_gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            roi_up = cv2.resize(
                roi_thresh, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )
            config = (
                "--psm 8 -c tessedit_char_whitelist=0123456789אבגדהוזחטיכלמנסעפצקרשת׳"
            )
            text = pytesseract.image_to_string(
                roi_up, lang="heb", config=config
            ).strip()
            text = re.sub(r"[^0-9א-ת״׳]", "", text)

            # Skip if text does not contain at least one digit
            if not re.search(r"\d", text):
                continue

            if debug:
                cv2.circle(image, (x, y), r, (255, 0, 0), 10)
                cv2.putText(
                    image,
                    text,
                    (x - r, y - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )

            if text:
                results.append(
                    {"center": (int(x), int(y)), "radius": int(r), "text": text}
                )

    if debug:
        cv2.imshow("Detected", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results
