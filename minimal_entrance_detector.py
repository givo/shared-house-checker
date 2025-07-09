import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class MinimalEntranceDetector:
    def __init__(self):
        # Tunable parameters
        self.arrow_area_min = 50
        self.arrow_area_max = 2000
        self.association_distance = 100  # pixels

    def preprocess_image(self, image):
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
        self, image, target_rgb, tolerance=10, padding=20
    ):
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

    def visualize_apartments(self, image, apartment_masks):
        for apt in apartment_masks:
            # Create colored overlay from mask
            mask = apt["mask"]
            overlay = image.copy()
            color = (0, 255, 0)  # Green highlight

            # Contour from mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # cv2.drawContours(overlay, contours, -1, color, 2)

            # Mark centroid
            c = tuple(map(int, apt["centroid"]))
            cv2.circle(overlay, c, 5, (255, 0, 0), -1)

            # Show result
            plt.figure(figsize=(6, 6))
            plt.imshow(overlay)
            plt.title(f"Apartment ID: {apt['id']} | Area: {apt['area']}")
            plt.axis("off")
            plt.show()

    def detect_apartment_entrances(self, image_path):
        """Main function to detect apartment entrances"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Step 1: Preprocessing
        processed_image = self.preprocess_image(image)

        # Step 2: Extract apartment by color (fuzzy match)
        target_color = (191, 126, 255)  # Example apartment color
        cropped, mask = self.extract_apartment_by_color_fuzzy(
            processed_image, target_color, tolerance=12, padding=20
        )

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
        self.visualize_apartments(cropped, apartment_masks)

        print("done")


# Usage example
if __name__ == "__main__":
    detector = MinimalEntranceDetector()

    # Replace with your image path
    image_path = os.path.join(os.path.dirname(__file__), "data", "part_of_floor_0.png")

    try:
        results, debug_info = detector.detect_apartment_entrances(image_path)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to provide a valid image path")
