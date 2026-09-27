# !/usr/bin/env python3
# shebang line specifies the interpreter to use

import cv2 as cv
import numpy as np

# events = [i for i in dir(cv) if "EVENT" in i]
# print(events)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),100,(255,0,0),-1)

# Create a black image, a window and bind the function to window
img = np.zeros((512,512, 3), np.uint8)
cv.namedWindow("image")
cv.setMouseCallback("image",draw_circle)

while(1):
    cv.imshow("image",img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()

