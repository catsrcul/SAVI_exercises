# !/usr/bin/env python3
# shebang line specifies the interpreter to use

import cv2 as cv
import numpy as np

# main function
def main():
    print('SAVI class')

    # absolute path
    # image = cv.imread()  # i dislike abs paths xd

    # Relative path
    image = cv.imread('images/cenario.jpg')

    print('shape:' + str(image.shape))
    print('dtype =' + str(image.dtype))

    cv.imshow("image", image)
    # cv.waitKey(0)

    # Get grayscale image
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    print("gray shape=" + str(gray_image))
    print("dtype =" + str(gray_image.dtype))
    cv.imshow("gray_image", gray_image)
    cv.waitKey(0)






if __name__== '__main__':
    main()
