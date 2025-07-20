import cv2
import os

import numpy as np


def preprocess_blueprint(
    image_path: str, upscale_factor: float = 2.0, denoise_strength: int = 10
) -> str:
    """
    Preprocesses a scanned blueprint image:
    - Upscales it to improve line clarity
    - Converts to grayscale
    - Denoises the image to remove paper noise
    - Enhances contrast using CLAHE
    - Smooths while preserving edges using bilateral filtering

    Args:
        image_path (str): Path to the input image
        upscale_factor (float): Factor to upscale the image (default: 2.0)
        denoise_strength (int): Strength of the denoising filter (default: 10)

    Returns:
        str: Path to the saved preprocessed image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Step 1: Upscale
    upscaled = cv2.resize(
        img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC
    )

    hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 2.0, 0, 255).astype(np.uint8)
    color_boosted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Step 2: Grayscale
    # gray = cv2.cvtColor(color_boosted, cv2.COLOR_BGR2GRAY)

    # Step 3: Denoise
    denoised = cv2.fastNlMeansDenoising(color_boosted, h=denoise_strength)

    # Step 4: Adaptive Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    r, g, b = cv2.split(color_boosted)
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)
    color_boosted = cv2.merge((r, g, b))

    # Step 5: Edge-Preserving Smoothing (Bilateral Filter)
    smoothed = cv2.bilateralFilter(color_boosted, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 6: Save result in same directory with "_preprocessed.jpg" suffix
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_preprocessed.jpg"
    cv2.imwrite(output_path, smoothed)

    return output_path


if __name__ == "__main__":
    # Example usage
    input_image_path = "/Users/givo/Pictures/b1.png"
    preprocessed_image_path = preprocess_blueprint(input_image_path)
    print(f"Preprocessed image saved at: {preprocessed_image_path}")
