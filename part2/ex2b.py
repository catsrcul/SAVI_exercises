#!/usr/bin/env python3
"""
Exercise 2b - Grass segmentation using the HSV color space.
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
    print("SAVI Exercise 2b - HSV Grass Segmentation")

    # Read and downsample image for convenient viewing
    image = cv.imread("images/dog_1.jpg")
    if image is None:
        print("Error: Could not load image 'images/dog_1.jpg'. Check the path.")
        return

    height, width = image.shape[:2]
    image = cv.resize(image, (round(width / 2), round(height / 2)))
    cv.imshow("Original", image)

    # -------------------------------------------------------------
    # 1. Conversion to HSV color space and channel inspection
    # -------------------------------------------------------------
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Split channels (H: 0-179 in OpenCV, S: 0-255, V: 0-255)
    h, s, v = cv.split(hsv_image)
    cv.imshow("H (Hue)", h)
    cv.imshow("S (Saturation)", s)
    cv.imshow("V (Value)", v)

    # -------------------------------------------------------------
    # 2. Segmenting grass using Hue (H) and Saturation (S)
    # -------------------------------------------------------------
    # Method A: Using OpenCV's cv.inRange (standard and fast)
    # Green grass in OpenCV typically spans Hue around 25-85 and moderate-to-high Saturation
    lower_grass = np.array([25, 40, 40], dtype=np.uint8)
    upper_grass = np.array([85, 255, 255], dtype=np.uint8)
    mask_grass = cv.inRange(hsv_image, lower_grass, upper_grass)

    show_mask("Grass Mask (inRange)", mask_grass)

    # Invert the grass mask to get the object (dog) candidate
    mask_dog = cv.bitwise_not(mask_grass)
    show_mask("Dog Candidate Mask", mask_dog)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
