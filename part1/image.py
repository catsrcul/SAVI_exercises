import cv2 as cv
import sys

img = cv.imread('frame_0016.png')

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)

k = cv.waitKey(0)   # Wait for a key press indefinitely, due to 0 passed as argument
                    # to end the program

if k == ord("s"):
    cv.imwrite("frame_0016.png", img)

