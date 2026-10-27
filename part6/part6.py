#! /usr/bin/env python3

import cv2 as cv
import numpy as np
import os
from scipy.optimize import least_squares

# 1. Ensure path exists
path = 'professor_main_part6/images/marvao/1.png'
if not os.path.exists(path):
    print(f"Error: file not found at {os.path.abspath(path)}")
# 2. Read image
# use cv.IMREAD_GRAYSCALE to work with gray scale
img = cv.imread(path)

img_float = img.astype(np.float64)


# 3. Verify object is not None before imshow
if img is not None:
    cv.imshow('Original Image', img)
    # cv.waitKey(0)  # Blocks execution until any key is pressed
    # cv.destroyAllWindows()
else:
    print("Could not decode image.")



def residual_brightness_only(params: np.ndarray, img: np.ndarray, target_mu: float) -> np.ndarray:
    s, b = params
    transformed = s * img + b
    return np.array([np.mean(transformed) - target_mu], dtype=np.float64)


def residual_brightness_contrast(params: np.ndarray, img: np.ndarray, target_mu: float,
                                 target_sigma: float) -> np.ndarray:
    s, b = params
    # 1. Apply affine model to image matrix
    transformed = s * img + b

    # 2. Compute current sample statistics
    current_mu = np.mean(transformed)
    current_sigma = np.std(transformed)

    # 3. Formulate residual vector r(θ)
    return np.array([current_mu - target_mu, current_sigma - target_sigma], dtype=np.float64)


# computing res
res = residual_brightness_only([1,1], img_float, 80)
print('the residual is ' + str(res))

solution = least_squares(residual_brightness_only, x0=[1.0, 0.0], args=(img_float, 80))
print('the solution is' + str(solution))


# Apply optimal parameters
s_opt, b_opt = solution.x
img_b = np.clip(s_opt * img_float + b_opt, 0, 255).astype(np.uint8)

# optimal for both color and contrast
solution_bc = least_squares(residual_brightness_contrast, x0=[1.0, 0.0], args=(img_float, 80, 60))
img_bc = np.clip(s_opt * img_float + b_opt, 0, 255).astype(np.uint8)

# Display result
cv.imshow('Corrected Image brightness', img_b)

cv.imshow('Corrected Image brightness and contrast', img_bc)

cv.waitKey(0)
cv.destroyAllWindows()