# !/usr/bin/env python3
# shebang line specifies the interpreter to use

import cv2 as cv
import numpy as np

# Relative path
image = cv.imread('images/dog_1.jpg')

copy_image = image.copy()

copy_image[:, :, 0] = 0
copy_image[:, :, 2] = 0

# cv.imshow('leaving only green', copy_image)

green_mask = cv.threshold(image[:, :, 1], 210, 255, cv.THRESH_BINARY)


cv.imshow('thresholding =', green_mask[1])

cut_image = cv.bitwise_and(image, image, mask=green_mask[1])
cv.imshow("cut image", cut_image)

# -----------------------------------------------------------------
# postprocessing using morphological operations
# we can dilate the pixels to eliminate unwanted parts of the image

kernel = np.ones((5,5), np.unit8)

mask_dilate = cv.dilate(green_mask, kernel, interations=2)


showMask('dilated mask', mask_dilate)

cv.waitKey(0)
cv.destroyAllWindows()
