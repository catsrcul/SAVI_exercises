#!/usr/bin/env python3
# Shebang line" specifies the interpreter.

import cv2 as cv
import numpy as np

def show_mask(window_name, image, show=0):
    image_to_show = image.astype(np.uint8)
    if show:
        cv.imshow(window_name, image_to_show)

# main functino
def main():
    # setup the video capture object





if __name__ == "__main__":
    main()