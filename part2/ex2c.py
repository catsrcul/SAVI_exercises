#!/usr/bin/env python3
"""
Exercise 2c - Morphological Operations & Mask Post-Processing.

Goal:
Take a noisy binary mask (from HSV segmentation) and clean it up
using morphological operations (erosion, dilation, opening, closing).
"""

import cv2 as cv
import numpy as np


def show_mask(window_name: str, mask: np.ndarray) -> None:
    """Displays a binary mask, properly scaled to [0, 255] if boolean."""
    if mask.dtype == bool:
        cv.imshow(window_name, (mask.astype(np.uint8) * 255))
    else:
        cv.imshow(window_name, mask)


def main():
    print("SAVI Exercise 2c - Morphological Post-Processing")

    # -------------------------------------------------------------
    # 0. Base Setup: Load image & obtain initial candidate mask
    # -------------------------------------------------------------
    image = cv.imread("images/dog_1.jpg")
    if image is None:
        print("Error: Could not load image 'images/dog_1.jpg'. Check the path.")
        return

    height, width = image.shape[:2]
    image = cv.resize(image, (round(width / 2), round(height / 2)))
    cv.imshow("Original", image)

    # Initial segmentation (HSV grass mask -> inverted to get dog candidate)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_grass = np.array([25, 40, 40], dtype=np.uint8)
    upper_grass = np.array([85, 255, 255], dtype=np.uint8)
    mask_grass = cv.inRange(hsv_image, lower_grass, upper_grass)

    # Raw mask containing the dog (with noise and holes)
    raw_dog_mask = cv.bitwise_not(mask_grass)
    show_mask("Raw Dog Mask (Before Morph)", raw_dog_mask)

    # =============================================================
    # WHERE TO START WRITING: STEP 1 GOES HERE
    # =============================================================

    # -------------------------------------------------------------
    # STEP 1: Creating the Structuring Element (Kernel)
    # -------------------------------------------------------------
    # Conceptual guidance:
    # A morphological filter slides a small neighborhood matrix (kernel)
    # over each pixel of the binary mask.
    #
    # INGREDIENTS:
    # - Shape & Size: Odd square dimensions are standard, e.g. (3, 3), (5, 5), (7, 7).
    #   Larger kernels affect a wider neighborhood on each pass.
    # - Data Type: Must be unsigned 8-bit integers (`np.uint8`).
    #
    # Two common ways to build a kernel:
    #   Method A (NumPy):
    #       np.ones((kernel_height, kernel_width), dtype=np.uint8)
    #   Method B (OpenCV):
    #       cv.getStructuringElement(shape, (kernel_width, kernel_height))
    #       where shape can be cv.MORPH_RECT, cv.MORPH_ELLIPSE, or cv.MORPH_CROSS.
    #
    # 1. Define your kernel dimensions (start with 5x5 or 7x7).
    # 2. Create the kernel variable using Method A or B.
    # 3. Print the kernel to the console to observe its matrix layout and dtype.
    #
    # Write your Step 1 code below:

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    # print("Kernel shape:", kernel.shape)
    # print("Kernel: \n", kernel)


    # -------------------------------------------------------------
    # STEP 2: Basic Operations (Erosion & Dilation)
    # -------------------------------------------------------------
    # Conceptual guidance:
    # - Erosion (`cv.erode`): A pixel remains 1 ONLY if all pixels under
    #   the kernel are 1. It shrinks white foreground areas.
    #   -> Great for removing small, isolated noise specks in the grass.
    #
    # - Dilation (`cv.dilate`): A pixel becomes 1 if ANY pixel under
    #   the kernel is 1. It expands white foreground areas.
    #   -> Great for bridging gaps and filling small holes inside the dog.
    #
    # INGREDIENTS:
    # - `cv.erode(src=mask, kernel=kernel, iterations=n)`
    # - `cv.dilate(src=mask, kernel=kernel, iterations=n)`
    #
    # 1. Try applying `cv.erode()` to `raw_dog_mask` and view it with `show_mask()`.
    # 2. Try applying `cv.dilate()` to `raw_dog_mask` and view it with `show_mask()`.
    # 3. Notice the trade-off: eroding removes noise but thins the dog;
    #    dilating fills holes but grows the noise!
    #
    # Write your Step 2 code below:

    eroded_mask = cv.erode(src=raw_dog_mask, kernel=kernel, iterations=3)
    dilated_mask = cv.dilate(src=eroded_mask, kernel=kernel, iterations=3)
    show_mask("eroded and dilated", dilated_mask)



    # -------------------------------------------------------------
    # STEP 3: Compound Operations (Opening & Closing)
    # -------------------------------------------------------------
    # Conceptual guidance:
    # - Opening (Erode -> Dilate): Removes small background noise points
    #   without significantly shrinking the main object.
    # - Closing (Dilate -> Erode): Fills internal holes and bridges breaks
    #   without significantly enlarging the outer boundary.
    #
    # INGREDIENTS:
    # - `cv.morphologyEx(src=mask, op=cv.MORPH_OPEN, kernel=kernel)`
    # - `cv.morphologyEx(src=mask, op=cv.MORPH_CLOSE, kernel=kernel)`
    #
    # 1. Apply Openiang to eliminate outer noise.
    # 2. Apply Closing to seal holes inside the dog.
    # 3. (Optional) Chain them together (e.g., Open then Close, or Close then Open).
    # 4. Display your cleaned final mask!
    #
    # Write your Step 3 code below:

    opened_mask = cv.morphologyEx(src=raw_dog_mask, op=cv.MORPH_OPEN, kernel=kernel, iterations=3)
    # closed_mask = cv.morphologyEx(src=oppened_mask, op=cv.MORPH_CLOSE, kernel=kernel, iterations=3)
    show_mask("mophologyEX", opened_mask)


    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
