#!/usr/bin/env python3
# Shebang line" specifies the interpreter.

# imports --------------------
import cv2
import numpy as np
from open3d.examples.pipelines.colored_icp_registration import source


def showMask(window_name, image):
    image_to_show = image.astype(np.uint8)*255
    cv2.imshow(window_name, image_to_show)

# Main function
def main(): # this is our main function
    print("SAVI exercise")

    # -------------------------------------------------------------------------
    # Setup the video capture object
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture("p3_assets/traffic.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps = ' + str(fps))

    fgbg = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=100, detectShadows=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    while True:

        # -------------------------------------------------------------------------
        # Read the frame
        # -------------------------------------------------------------------------
        ret, image_rgb = cap.read()
        if not ret: # test to see if the video reading is finsished
            print('Completed processing video')
            break

        cv2.imshow('Image', image_rgb) # Display the resulting frame

        fg_mask = fgbg.apply(image_rgb)

        opened_mask = cv2.morphologyEx(src=fg_mask, op=cv2.MORPH_OPEN, kernel=kernel, iterations=4)

        cv2.imshow('foreground mask natural', fg_mask)
        cv2.imshow("foreground_mask", opened_mask)

        # no_background_video = image_rgb

        # -------------------------------------------------------------------------
        # How to detect the cars passing through the lanes
        # -------------------------------------------------------------------------
        # How many cars passed through lane X?
        # Colors of the cars




        # -------------------------------------------------------------------------
        # handle key press events
        # -------------------------------------------------------------------------
        key = cv2.waitKey(50)
        if key == 113:
            print('Pressed q. Aborting ')
            break

if __name__ == "__main__":
    main()