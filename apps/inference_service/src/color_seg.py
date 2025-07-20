import os
import cv2
import numpy as np
import sys
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

# Parameters
TOLERANCE = 15  # Color distance tolerance in LAB
DILATION_ITERS = 3  # How much to expand the region


def extract_color_mask(image_bgr, target_bgr, tolerance=30, dilation_iters=3):
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    target_bgr = np.array([[target_bgr]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Step 1: Euclidean distance in LAB space
    dist = np.linalg.norm(
        image_lab.astype(np.float32) - target_lab.astype(np.float32), axis=2
    )
    # dist = deltaE_cie76(image_lab, np.full_like(image_lab, target_lab))
    initial_mask = (dist < tolerance).astype(np.uint8) * 255

    # Step 2: Morphological expansion
    kernel = np.ones((10, 10), np.uint8)
    expanded_mask = cv2.dilate(initial_mask, kernel, iterations=dilation_iters)
    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, kernel)

    return initial_mask, expanded_mask


def get_skeleton_endpoints(mask, contour):
    # Create a blank mask for this contour only
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Skeletonize
    skeleton = skeletonize(contour_mask > 0).astype(np.uint8)

    # Find endpoints: pixels with exactly one neighbor
    endpoints = []
    h, w = skeleton.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x]:
                neighbors = skeleton[y - 1 : y + 2, x - 1 : x + 2]
                if np.sum(neighbors) == 2:  # itself + 1 neighbor
                    endpoints.append((x, y))

    if len(endpoints) < 2:
        return []

    # Find the two farthest endpoints
    endpoints = np.array(endpoints)
    dists = cdist(endpoints, endpoints)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return [tuple(endpoints[i]), tuple(endpoints[j])]


def find_valid_seed_point(mask, contour, search_stride=5):
    x, y, w, h = cv2.boundingRect(contour)
    h_mask, w_mask = mask.shape

    for dy in range(0, h, search_stride):
        for dx in range(0, w, search_stride):
            px = x + dx
            py = y + dy
            if px >= w_mask or py >= h_mask:
                continue

            inside = cv2.pointPolygonTest(contour, (px, py), False)
            if inside >= 0 and mask[py, px] != 0:
                return (px, py)

    return None


def flood_fill_inside_binary_mask(mask, seed_point):
    h, w = mask.shape
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Mark the white regions from the mask as fillable (value 1 in OpenCV floodFill)
    fill_mask[1 : h + 1, 1 : w + 1] = (mask > 0).astype(np.uint8)

    # Dummy image — we just need it to run the fill
    dummy_img = np.zeros((h, w), np.uint8)

    # Run the fill — will only grow inside areas where fill_mask == 1
    cv2.floodFill(
        image=dummy_img,
        mask=fill_mask,
        seedPoint=seed_point,
        newVal=255,
        flags=cv2.FLOODFILL_MASK_ONLY | 4,
    )

    # The filled region will now be marked as 1 in the mask
    result = (fill_mask[1:-1, 1:-1] == 1).astype(np.uint8) * 255
    return result


def get_open_endpoints(contour):
    if len(contour) < 2:
        return []

    # Flatten the contour
    pts = contour[:, 0, :]  # shape (N, 2)

    # Find the two farthest points (brute force)
    max_dist = 0
    pt1, pt2 = pts[0], pts[1]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist:
                max_dist = d
                pt1, pt2 = pts[i], pts[j]

    return [tuple(pt1), tuple(pt2)]


def bridge_open_contours(mask, debug_image, max_distance=300):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    endpoints = []  # list of (point, contour_index)
    for idx, contour in enumerate(contours):
        eps = get_skeleton_endpoints(mask, contour)
        for pt in eps:
            endpoints.append((pt, idx))

    # Debug draw
    if debug_image is not None:
        for pt, _ in endpoints:
            cv2.circle(debug_image, pt, 3, (0, 0, 255), -1)
        cv2.imshow("Skeleton Endpoints", debug_image)

    used = set()
    bridge_lines = []

    for i, (p1, c1) in enumerate(endpoints):
        if i in used:
            continue

        best_match = None
        best_dist = float("inf")

        for j, (p2, c2) in enumerate(endpoints):
            if i == j or j in used or c1 == c2:
                continue

            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < best_dist and dist < max_distance:
                best_dist = dist
                best_match = (j, p2)

        if best_match:
            j, p2 = best_match
            cv2.line(mask, p1, p2, 255, 1)
            used.add(i)
            used.add(j)

    return mask


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        image = param["image"]
        color = image[y, x].tolist()
        yellow = [171, 247, 252]
        pink = [162, 165, 249]
        purple = (199, 158, 197)
        color = purple
        print(f"Clicked color (BGR): {color}")

        initial_mask, expanded_mask = extract_color_mask(
            image, color, TOLERANCE, DILATION_ITERS
        )

        # Show initial and expanded masks
        cv2.imshow("Initial Color Match", initial_mask)
        cv2.imshow("Expanded Mask", expanded_mask)

        # Find contours
        # contours, _ = cv2.findContours(
        #     expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # Step: Stitch broken contours together
        debug_img = expanded_mask.copy()
        debug_img = cv2.cvtColor(expanded_mask.copy(), cv2.COLOR_GRAY2BGR)
        bridged_mask = bridge_open_contours(expanded_mask, debug_image=debug_img)

        # Show bridged mask for debug
        cv2.imshow("Bridged Mask", bridged_mask)

        # Find contours on repaired mask
        contours, _ = cv2.findContours(
            bridged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            biggest = max(contours, key=cv2.contourArea)
            seed_point = find_valid_seed_point(expanded_mask, biggest)

            if seed_point:
                print(f"Seed point for fill: {seed_point}")
                filled_mask = flood_fill_inside_binary_mask(expanded_mask, seed_point)

                if filled_mask is not None:
                    # Overlay on original
                    overlay = image.copy()
                    overlay[filled_mask > 0] = [0, 255, 0]
                    cv2.circle(overlay, seed_point, 4, (0, 0, 255), -1)  # mark seed
                    cv2.imshow("Photoshop-style Fill", overlay)

                    # Crop based on fill region
                    contours_fill, _ = cv2.findContours(
                        filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours_fill:
                        x, y, w, h = cv2.boundingRect(contours_fill[0])
                        cropped = image[y : y + h, x : x + w]
                        cv2.imshow("Cropped Region", cropped)
                else:
                    print("❌ Flood fill failed.")
            else:
                print("❌ No valid seed point found inside contour.")


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        sys.exit(1)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", on_mouse_click, {"image": image})

    print("Click on a color in the image to extract its mask...")

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), "data", "hard_floor.png")
    main(image_path)
