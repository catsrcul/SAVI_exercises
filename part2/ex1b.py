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

    # cv.imshow("image", image)
    # cv.waitKey(0)

    # Get grayscale image
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    print("gray shape=" + str(gray_image.shape))
    print("dtype =" + str(gray_image.dtype))
    # cv.imshow("gray_image", gray_image)
    # cv.waitKey(0)

    # How to get a portion of the image
    left_side_image = image[:, :image.shape[1]//2]
    # cv.imshow("left_side_image", left_side_image)
    right_side_image = gray_image[:, gray_image.shape[1]//2:]
    # cv.imshow("right_side_image", right_side_image)

    # convert 1 channel gray image to 3 channels for compatibility
    right_side_image = cv.cvtColor(right_side_image, cv.COLOR_GRAY2BGR)

    connected_image = cv.hconcat([left_side_image, right_side_image])
    cv.imshow("connected_image", connected_image)

    # resizing images
    resize_image = cv.resize(gray_image, (256, 512))
    cv.imshow("resize_image", resize_image)

    # image brightness increase can cause overflow
    # Convert to int16/float, add 20, clip between 0 and 255, and convert back to uint8
    bright_gray_np = np.clip(gray_image.astype(np.int16) + 100, 0, 1000).astype(np.uint8)
    cv.imshow("Bright Gray (NumPy clip)", bright_gray_np)

    # cv.add fixes this by maxing out the value at 255
    bright_gray = cv.add(gray_image, 100)
    cv.imshow("bright_gray", bright_gray)

    # doing this while converting the values to float
    gray_float = gray_image.astype(np.float32)

    gray_float = gray_float + 20

    # a image expects a uint8 value so before we need to convert the values
    bright_gray = np.clip(gray_float, 0, 255).astype(np.uint8)

    # cv.imshow("float image", bright_gray)

    # Wait for a key press so windows stay open
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
