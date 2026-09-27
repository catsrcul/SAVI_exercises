#!/usr/bin/env python3
# Shebang line" specifies the interpreter.

# imports --------------------
import cv2
import numpy as np


def safe_alter_image(image, curtain_x, value=-100):
    # alter only the left side

    image_float = image.astype(float)
    height, width, nc = image.shape

    # safe brigthen the image
    altered_image = image_float.copy()

    # creating a fraction of the column
    col_limit = round(width * curtain_x)

    altered_image[:, 0:col_limit, :] = image_float[:, 0:col_limit, :] + value


    # some elements will have values over 255,
    # so we need to clip the values to 255
    altered_image = altered_image.clip(0, 255)

    # convert back to uint8
    altered_image = altered_image.astype(np.uint8)

    return altered_image


# Main function
def main():  # this is our main function
    print("SAVI exercise")

    # relative path
    image = cv2.imread("images/dog_1.jpg")
    height, width, channels = image.shape
    image = cv2.resize(image, (round(width / 2), round(height / 2)))
    cv2.imshow('Original', image)
    H, W, NC = image.shape

    # --------------------------------
    # Darken the image
    # --------------------------------
    altered_image = safe_alter_image(image, curtain_x=1/3 , value=-50)
    cv2.imshow("Altered Image  ", altered_image)

    # Create a sequence of progressive darkening of the left side of the image
    for i in range(0, 20):
        value_i = -i * 10
        curtain_x = 0.05 + i * 0.05
        altered_image = safe_alter_image(image, curtain_x, value_i)
        cv2.imshow("Altered Image  ", altered_image)
        cv2.waitKey(500)

    # --------------------------------
    # Challenge for homework
    # --------------------------------

    # a) record a video of the sequence

    # b)  make the darkened color be a curtain that moved from left to right

    cv2.waitKey(0)


if __name__ == "__main__":
    main()