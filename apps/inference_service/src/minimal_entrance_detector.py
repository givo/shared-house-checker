import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional


class MinimalEntranceDetector:
    def __init__(self):
        # Tunable parameters
        self.arrow_area_min = 40  # from 50
        self.arrow_area_max = 2500  # from 2000
        self.association_distance: int = 100  # pixels

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Basic preprocessing - convert to RGB and enhance contrast"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return enhanced

    def extract_apartment_by_color_fuzzy(
        self,
        image: np.ndarray,
        target_rgb: Tuple[int, int, int],
        tolerance: int = 10,
        padding: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the largest region made of visually connected pixels of a certain color,
        robust to thin black blueprint lines interrupting walls.
        """
        image = np.array(image)

        # Fuzzy match on color
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lower = np.clip(np.array(target_rgb) - tolerance, 0, 255)
        upper = np.clip(np.array(target_rgb) + tolerance, 0, 255)
        lower_bgr = np.array([lower[2], lower[1], lower[0]])
        upper_bgr = np.array([upper[2], upper[1], upper[0]])
        mask = cv2.inRange(image_bgr, lower_bgr, upper_bgr)

        # Close small gaps caused by black lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Optional dilation to make sure thin sections stay connected
        dilated = cv2.dilate(closed_mask, np.ones((3, 3), np.uint8), iterations=1)

        # Connected components on clean mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )

        if num_labels <= 1:
            raise ValueError("No connected apartment-like region found.")

        # Pick largest non-background region
        best_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h = stats[best_index][:4]

        # Padding
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(image.shape[1], x + w + padding)
        y_max = min(image.shape[0], y + h + padding)

        # Final crop
        cropped = image[y_min:y_max, x_min:x_max]
        final_mask = (labels == best_index).astype(np.uint8) * 255
        cropped_mask = final_mask[y_min:y_max, x_min:x_max]

        return cropped, cropped_mask

    def visualize_apartments(
        self, image: np.ndarray, apartment_masks: List[Dict[str, Any]]
    ) -> None:
        for apt in apartment_masks:
            # Create colored overlay from mask
            mask = apt["mask"]
            overlay = image.copy()
            color = (0, 255, 0)  # Green highlight

            # Contour from mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Mark centroid
            c = tuple(map(int, apt["centroid"]))
            cv2.circle(overlay, c, 5, (255, 0, 0), -1)

            # Show result
            plt.figure(figsize=(6, 6))
            plt.imshow(overlay)
            plt.title(f"Apartment ID: {apt['id']} | Area: {apt['area']}")
            plt.axis("off")
            plt.show()

    def detect_arrows(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect black arrow shapes"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold for black arrows
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations to enhance arrow shapes
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        arrows: List[Dict[str, Any]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.arrow_area_min or area > self.arrow_area_max:
                continue

            approx = cv2.approxPolyDP(
                contour, 0.03 * cv2.arcLength(contour, True), True
            )
            if len(approx) < 6 or len(approx) > 8:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / (h + 1e-5)
            if aspect_ratio < 1.0:  # Only horizontal-ish arrows
                continue

            # Moments and direction estimation
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

            arrows.append(
                {
                    "position": np.array([cx, cy]),
                    "direction": direction,
                    "contour": contour,
                    "area": area,
                }
            )

        return arrows

    def visualize_arrows(self, image: np.ndarray, arrows: List[Dict[str, Any]]) -> None:
        overlay = image.copy()

        for arrow in arrows:
            cx, cy = arrow["position"].astype(int)

            # Draw a red bounding circle or box
            cv2.circle(overlay, (cx, cy), 10, (255, 0, 0), 2)  # Red circle

            # Optionally, draw direction vector
            dx, dy = (arrow["direction"] * 20).astype(int)
            tip = (cx + dx, cy + dy)
            cv2.arrowedLine(overlay, (cx, cy), tip, (0, 0, 255), 2, tipLength=0.3)

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"Detected Arrows: {len(arrows)}")
        plt.axis("off")
        plt.show()

    def extract_and_draw_apartment_contour(self, image: np.ndarray):
        """
        Given a cropped apartment image, extract the outer contour and draw it in green.

        Args:
            image (np.ndarray): Cropped RGB or BGR image of apartment region.

        Returns:
            np.ndarray: The image with the detected contour drawn in green.
        """
        # Step 1: Preprocess image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 2: Edge detection
        edges = cv2.Canny(blurred, threshold1=25, threshold2=75)

        # Step 3: Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("No contours found.")
            return image

        # Step 4: Select the largest contour (assumes it's the outer wall)
        largest_contour = max(contours, key=cv2.contourArea)

        # Optional: simplify the contour (remove spikes)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Step 5: Draw the contour on the original image (in green)
        output = image.copy()
        cv2.drawContours(output, [approx], -1, (0, 255, 0), thickness=2)

        cv2.imshow("Contour", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_apartment_entrances(
        self, image_path: str
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Main function to detect apartment entrances"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Step 1: Preprocessing
        processed_image = self.preprocess_image(image)

        # Step 2: Extract apartment by color (fuzzy match)
        purple = (191, 126, 255)
        brown = (221, 166, 110)
        green = (0, 150, 0)
        target_color = brown
        cropped, mask = self.extract_apartment_by_color_fuzzy(
            processed_image, target_color, tolerance=30, padding=20
        )

        # ✅ Save the cropped image
        save_path = os.path.join(
            os.path.dirname(image_path), "cropped_apartment_brown.jpg"
        )
        cv2.imwrite(save_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        # Step 3: Detect arrows
        # arrows = self.detect_arrows(cropped)

        # Step 4: Visualize apartment + arrows
        # self.visualize_arrows(cropped, arrows)

        # Create fake mask data structure to visualize
        apartment_masks = [
            {
                "mask": mask,
                "centroid": np.array([mask.shape[1] // 2, mask.shape[0] // 2]),
                "area": np.count_nonzero(mask),
                "id": 1,
            }
        ]
        # Visualize just this one apartment
        self.extract_and_draw_apartment_contour(cropped)
        self.visualize_apartments(cropped, apartment_masks)

        print("done")
        return None, None


# Usage example
if __name__ == "__main__":
    detector = MinimalEntranceDetector()

    # image_path = os.path.join(os.path.dirname(__file__), "data", "floor1.jpg")
    image_path = os.path.join(os.path.dirname(__file__), "data", "part_of_floor_0.png")

    try:
        results, debug_info = detector.detect_apartment_entrances(image_path)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to provide a valid image path")
