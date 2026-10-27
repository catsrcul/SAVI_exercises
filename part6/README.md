# SAVI - Part 6: Optimization Applied to Computer Vision (Photometry and Geometric Alignment)

This document provides the theoretical background, mathematical formulations, OpenCV / SciPy / NumPy API signatures, and isolated code snippets needed to solve the problems in **Part 6**.

---

## 1. Exercise 1: Photometric Correction via Target Mean Brightness and Contrast

### Theoretical & Mathematical Background

A radiometric / affine intensity transformation applied to an image $I(x, y)$ is parameterized by a scale factor (gain/contrast) $s \in \mathbb{R}$ and an offset (_bias_/brightness) $b \in \mathbb{R}$:

$$I'(x, y) = s \cdot I(x, y) + b$$

For an image with $N$ pixels, the key statistical metrics are:
- **Mean Brightness ($\mu$):**
  $$\mu(I') = \frac{1}{N} \sum_{x, y} I'(x, y)$$
- **Contrast ($\sigma$, standard deviation):**
  $$\sigma(I') = \sqrt{\frac{1}{N} \sum_{x, y} \left( I'(x, y) - \mu(I') \right)^2}$$

Given a target mean brightness $\mu_{\text{target}}$ and target contrast $\sigma_{\text{target}}$, the residual vector $\mathbf{r}(s, b)$ for nonlinear least squares is formulated as:

$$\mathbf{r}(s, b) = \begin{bmatrix} \mu(I') - \mu_{\text{target}} \\ \sigma(I') - \sigma_{\text{target}} \end{bmatrix}$$

The nonlinear least squares estimator minimizes the scalar cost function:
$$f(s, b) = \frac{1}{2} \|\mathbf{r}(s, b)\|_2^2 = \frac{1}{2} \left[ (\mu(I') - \mu_{\text{target}})^2 + (\sigma(I') - \sigma_{\text{target}})^2 \right]$$

> **Theoretical Note:** If only the brightness error is considered ($\mathbf{r} = [\mu(I') - \mu_{\text{target}}]$), the system has 1 constraint and 2 unknowns ($s, b$), leading to an infinite family of degenerate solutions. Adding the standard deviation constraint isolates a unique solution.

### API & Mechanical Operation

- `scipy.optimize.least_squares(fun, x0, args=(), bounds=(-inf, inf), method='trf', ...)`
  - `fun`: Objective function `f(params, *args) -> np.ndarray` returning a 1D vector of residuals.
  - `x0`: Initial parameter estimates (1D array).
  - Returns: An `OptimizeResult` object with the `.x` attribute containing optimal estimated parameters.
- Floating-point operations in NumPy:
  - `np.mean(a, axis=None, dtype=np.float64)`
  - `np.std(a, axis=None, dtype=np.float64)`
  - `np.clip(a, a_min, a_max)`: Clamps values to the physical range $[0, 255]$ after applying affine transforms.

### Minimal Isolated Code Snippet

```python
import numpy as np
from scipy.optimize import least_squares

def residual_brightness_contrast(params: np.ndarray, img_gray: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    s, b = params[0], params[1]
    # Apply affine model
    transformed = s * img_gray + b
    
    # Compute statistical metrics
    current_mean = np.mean(transformed)
    current_std = np.std(transformed)
    
    # Return 1D residual vector to the optimizer
    return np.array([current_mean - target_mean, current_std - target_std], dtype=np.float64)

# Synthetic invocation example:
dummy_img = np.random.uniform(50, 180, size=(100, 100)).astype(np.float64)
x0 = np.array([1.0, 0.0])  # s=1, b=0
res = least_squares(residual_brightness_contrast, x0, args=(dummy_img, 80.0, 60.0))
s_opt, b_opt = res.x
```

---

## 2. Exercise 2 & 3: Mosaic Color Correction (Photometric Alignment)

### Theoretical & Mathematical Background

Given two overlapping images, a reference train image $I_t(x, y)$ and a spatially warped query image $I_q(x, y)$:
- The overlap binary mask $\Omega$ is defined as:
  $$\Omega = \{ (x, y) \mid I_t(x, y) \text{ is valid} \land I_q(x, y) \text{ is valid} \}$$
  (e.g., where warped query pixel values are non-zero).

#### Exercise 2 (Pixel-wise Similarity):
The goal is to minimize photometric differences across the overlap region $\Omega$:
$$r_{xy}(s, b) = (s \cdot I_q(x, y) + b) - I_t(x, y), \quad \forall (x, y) \in \Omega$$

- **Degeneracy Issue:** Optimizing both $I_t$ parameters ($s_t, b_t$) and $I_q$ parameters ($s_q, b_q$) simultaneously without an absolute anchor yields the trivial global minimizer $s_t = s_q = 0$ and $b_t = b_q = 0$, rendering both images pure black with zero error ($0 - 0 = 0$). Keeping $I_t$ fixed prevents degeneracy.
- **Scalar vs. Vector Residuals:** Returning the scalar mean error $\frac{1}{|\Omega|} \sum r_{xy}$ allows large positive and negative residuals to cancel out. Returning the full 1D residual vector allows Gauss-Newton / Levenberg-Marquardt algorithms to compute the Jacobian matrix and properly converge.

#### Exercise 3 (Multi-Objective Weight Balancing):
Combining pixel-wise residuals with global brightness and contrast constraints introduces dimensional imbalance: there are $M = |\Omega| \gg 10^5$ pixel residuals but only 4 global statistical residuals. Since least squares minimizes $\sum_i r_i^2$, unweighted statistical residuals become negligible.

To balance their influence, introduce a weighting factor $w$:
$$w = \sqrt{\frac{|\Omega|}{K}}$$
where $K$ is the number of statistical terms (e.g., 2 or 4), equalizing variance between both loss components.

### API & Mechanical Operation

- Boolean indexing in NumPy to isolate the overlap region:
  `mask = (img_warped > 0)`
  `vals_q = img_warped[mask]`
  `vals_t = img_train[mask]`

### Minimal Isolated Code Snippet

```python
import numpy as np
from scipy.optimize import least_squares

def residual_similarity(params: np.ndarray, q_vals: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    s, b = params[0], params[1]
    # Pixel-to-pixel residual over valid overlap points
    return (s * q_vals + b) - t_vals

# 1D arrays of pixel values inside the overlap mask
q_overlap = np.array([50.0, 100.0, 150.0], dtype=np.float64)
t_overlap = np.array([70.0, 120.0, 170.0], dtype=np.float64)

x0 = np.array([1.0, 0.0])
res = least_squares(residual_similarity, x0, args=(q_overlap, t_overlap))
```

---

## 3. Exercise 4: Geometric Alignment via SIFT Features and 2D Optimization

### Theoretical & Mathematical Background

Assuming the geometric relationship between $I_q$ and $I_t$ is a pure 2D translation parameterized by $\mathbf{t} = [\Delta x, \Delta y]^T$:
$$\begin{bmatrix} x_t \\ y_t \end{bmatrix} = \begin{bmatrix} x_q \\ y_q \end{bmatrix} + \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}$$

Given a set of $P$ matched keypoint pairs $\{(\mathbf{p}_{q,i}, \mathbf{p}_{t,i})\}_{i=1}^P$:
The reprojection residual vector for the $i$-th matched pair is:
$$\mathbf{r}_i(\Delta x, \Delta y) = \begin{bmatrix} (x_{q,i} + \Delta x) - x_{t,i} \\ (y_{q,i} + \Delta y) - y_{t,i} \end{bmatrix}$$

The total least-squares objective function is:
$$E(\Delta x, \Delta y) = \sum_{i=1}^P \left[ (x_{q,i} + \Delta x - x_{t,i})^2 + (y_{q,i} + \Delta y - y_{t,i})^2 \right]$$

Setting the partial derivative with respect to $\Delta x$ to zero:
$$\frac{\partial E}{\partial \Delta x} = 2 \sum_{i=1}^P (x_{q,i} + \Delta x - x_{t,i}) = 0 \implies \Delta x^* = \frac{1}{P} \sum_{i=1}^P (x_{t,i} - x_{q,i})$$
This proves analytically why the optimal least-squares translation is identical to the arithmetic mean of coordinate differences across all matched pairs.

### API & Mechanical Operation

- `cv.SIFT_create()`
  - `sift.detectAndCompute(image, mask=None) -> (keypoints, descriptors)`
    - `keypoints`: List of `cv.KeyPoint` objects, where `.pt = (x, y)`.
    - `descriptors`: `np.ndarray` of type `float32` and shape `(N_pts, 128)`.
- `cv.BFMatcher(normType=cv.NORM_L2, crossCheck=False)`
  - `matcher.knnMatch(queryDescriptors, trainDescriptors, k=2) -> list[list[cv.DMatch]]`
- `cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, outImg, ...)`

### Minimal Isolated Code Snippet

```python
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

# 1. SIFT Extraction and Matching
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_L2)
raw_matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test filtering
good_pts1, good_pts2 = [], []
for m, n in raw_matches:
    if m.distance < 0.75 * n.distance:
        good_pts1.append(kp1[m.queryIdx].pt)
        good_pts2.append(kp2[m.trainIdx].pt)

pts_q = np.float32(good_pts1)  # Shape: (N, 2)
pts_t = np.float32(good_pts2)  # Shape: (N, 2)

# 2. 2D Translation Residuals
def residual_translation(delta: np.ndarray, p_query: np.ndarray, p_train: np.ndarray) -> np.ndarray:
    # delta = [dx, dy]
    p_trans = p_query + delta
    diff = p_trans - p_train
    return diff.flatten()  # 1D vector of length 2*N

res = least_squares(
    residual_translation,
    x0=np.array([0.0, 0.0]),
    args=(pts_q, pts_t),
    bounds=([-500, -500], [500, 500])
)
dx_opt, dy_opt = res.x
```

---

## 4. Exercise 5: Outlier Rejection via RANSAC and Robust Loss Functions

### Theoretical & Mathematical Background

Standard ordinary least squares assumes normally distributed errors. False matches (_outliers_) severely bias the estimation because quadratic penalties grow quadratically: $\mathcal{O}(r^2)$.

To handle outlier contamination:
1. **RANSAC (Random Sample Consensus):** An iterative algorithm that generates hypotheses from minimal sample sets and scores them by the number of inliers whose reprojection residual is within a threshold $\tau$.
2. **M-Estimators (Robust Loss Functions in `least_squares`):**
   - **Soft L1 (`loss='soft_l1'`):** Replaces the squared loss $\rho(z) = z$ with:
     $$\rho(z) = 2(\sqrt{1 + z} - 1), \quad \text{where } z = r^2$$
     For small residuals ($z \to 0$), it behaves like $L_2$ ($\rho(z) \approx z$). For large residuals ($z \to \infty$), its gradient scales down to $\frac{d\rho}{dz} \approx \frac{1}{\sqrt{z}}$, bounding the influence of severe outliers.

### API & Mechanical Operation

- `cv.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold=...)`
  - `method`: `cv.RANSAC` or `cv.USAC_MAGSAC`.
  - `ransacReprojThreshold`: Maximum reprojection error in pixels to consider a point an inlier.
  - Returns: `(H, mask)` where `mask` is an inlier mask (1 for inliers, 0 for outliers).
- `loss` parameter in `least_squares`:
  - `least_squares(..., loss='soft_l1', f_scale=1.0)`: `f_scale` sets the boundary between the linear regime and the attenuated outlier regime.

### Minimal Isolated Code Snippet

```python
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

# RANSAC filtering
H, inlier_mask = cv.findHomography(pts_q, pts_t, cv.RANSAC, ransacReprojThreshold=5.0)

# Robust least squares optimization
res_robust = least_squares(
    residual_translation,
    x0=np.array([0.0, 0.0]),
    args=(pts_q, pts_t),
    loss='soft_l1',
    f_scale=2.0
)
```

---

## 5. Exercise 6: Joint Direct Geometric and Photometric Alignment (Pixel-Based)

### Theoretical & Mathematical Background

In direct methods (without feature extraction), optimization minimizes intensity discrepancies directly over all pixels under spatial translation $(\Delta x, \Delta y) \in \mathbb{R}^2$ and photometric scaling/offset $(s, b) \in \mathbb{R}^2$:

$$e(x, y) = s \cdot I_q(x, y) + b - I_t(x + \Delta x, y + \Delta y), \quad \forall (x, y) \in \text{dom}(I_q)$$

The joint parameter vector to estimate is:
$$\mathbf{\theta} = [\Delta x, \Delta y, s, b]^T$$

#### Sub-Pixel Bilinear Interpolation:
Because $(\Delta x, \Delta y)$ is continuous in $\mathbb{R}$, sampled coordinates in the train image $(x + \Delta x, y + \Delta y)$ do not fall precisely onto discrete pixel centers. Bilinear interpolation via coordinate remapping is necessary.

#### Non-Convex Landscape & Local Minima:
Intensity-based pixel error surfaces are highly non-convex, featuring dense local minima from high-frequency textures. To guarantee convergence:
1. **Gaussian Multi-Scale Pyramids:** Downsample images (e.g. 1/8 scale) and apply Gaussian blur (`cv.GaussianBlur`), filtering out high frequencies to broaden the global basin of attraction.
2. **Initial Estimate:** Use template matching (`cv.matchTemplate`) to establish an initial integer estimate $\Delta x_0, \Delta y_0$.

### API & Mechanical Operation

- `cv.remap(src, map1, map2, interpolation, borderMode=..., borderValue=...)`
  - `map1`: Float32 matrix containing source $x$ coordinates to sample.
  - `map2`: Float32 matrix containing source $y$ coordinates to sample.
  - `interpolation`: `cv.INTER_LINEAR` for bilinear interpolation.
- `cv.matchTemplate(image, templ, method)`
  - `method`: `cv.TM_CCOEFF_NORMED` (range $[-1, 1]$, peak at 1).
- `cv.minMaxLoc(src)`
  - Returns: `(min_val, max_val, min_loc, max_loc)`.

### Minimal Isolated Code Snippet

```python
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

def sample_with_translation(img: np.ndarray, dx: float, dy: float, out_shape: tuple) -> np.ndarray:
    h, w = out_shape[:2]
    # Build translated coordinate sampling grids
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + np.float32(dx)
    map_y = grid_y + np.float32(dy)
    # Bilinear remapping
    warped = cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    return warped

def residual_joint(params: np.ndarray, img_q: np.ndarray, img_t: np.ndarray) -> np.ndarray:
    dx, dy, s, b = params
    # Sample train image at shifted query pixel locations
    t_sampled = sample_with_translation(img_t, dx, dy, img_q.shape)
    
    # Affine radiometric correction on query image
    q_transformed = s * img_q + b
    
    # Valid overlap mask
    valid_mask = (t_sampled > 0)
    
    diff = q_transformed[valid_mask] - t_sampled[valid_mask]
    return diff.flatten()

# Global initial estimate using Template Matching:
res_tm = cv.matchTemplate(img_train_small, img_query_small, cv.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv.minMaxLoc(res_tm)
init_dx, init_dy = max_loc

# Continuous sub-pixel optimization:
x0 = np.array([init_dx, init_dy, 1.0, 0.0], dtype=np.float64)
opt_res = least_squares(residual_joint, x0, args=(img_query_small, img_train_small))
```
