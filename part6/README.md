# SAVI - Part 6: Optimization in Computer Vision (Theory, Engineering Intuition & Mechanics)

Welcome to Part 6! This guide bridges rigorous mathematical formulations with engineering intuition and practical code mechanics.

---

## 0. The Big Picture: Optimization in Computer Vision

In mechanical systems and control theory, you often design a controller to minimize a cost or error function (e.g., tracking error, energy loss).

In computer vision, **optimization** follows the exact same principle:
1. **Model Parameterization:** We define a vector of unknown parameters $\mathbf{\theta} \in \mathbb{R}^P$ that governs a transformation (e.g., gains, spatial offsets).
2. **Residual Vector:** We define an objective mapping $\mathbf{r}(\mathbf{\theta}) \in \mathbb{R}^M$ measuring the discrepancy between our model's current output and the expected/reference target.
3. **Nonlinear Least Squares Solver:** An iterative algorithm (such as Levenberg-Marquardt or Trust Region Reflective, via `scipy.optimize.least_squares`) minimizes the sum of squared residuals:

$$\min_{\mathbf{\theta}} \frac{1}{2} \|\mathbf{r}(\mathbf{\theta})\|_2^2 = \min_{\mathbf{\theta}} \frac{1}{2} \sum_{i=1}^M r_i(\mathbf{\theta})^2$$

At each iteration $k$, the solver approximates the local landscape using a Taylor series and the Jacobian matrix $\mathbf{J} = \frac{\partial \mathbf{r}}{\partial \mathbf{\theta}}$, updating parameters via a step $\mathbf{\Delta \theta}$:

$$(\mathbf{J}^T \mathbf{J} + \lambda \mathbf{I}) \mathbf{\Delta \theta} = -\mathbf{J}^T \mathbf{r}$$

```
                ┌──────────────────────────────┐
                │ Initial Parameter Guess: θ₀  │
                └──────────────┬───────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ Compute Predictions & Residual Vector r(θ)   │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  Evaluate Stopping Criteria: ||r(θ)|| < ε    │
        └──────────────┬───────────────────────────────┘
                       │
             ┌─────────┴─────────┐
             │                   │
          NO ▼                YES▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ Calculate Jacobian J    │  │ Converged!              │
│ Take step: θ ← θ + Δθ   │  │ Return Optimal θ*        │
└────────────┬────────────┘  └─────────────────────────┘
             │
             └────────► (Next iteration)
```

---

## 1. Exercise 1: Photometric Control (Mean Brightness & Contrast)

### 1.1 The Physical & Mathematical Model
Every grayscale image $I(x, y)$ is a 2D intensity field. When captured by different sensors or under varying illumination, the received irradiance undergoes radiometric scaling and bias. We model this affine transformation pixel-by-pixel:

$$I'(x, y) = s \cdot I(x, y) + b$$

Where:
- $s \in \mathbb{R}^+$ is the **scale factor (gain/contrast)**. It expands ($s > 1$) or compresses ($s < 1$) the intensity histogram.
- $b \in \mathbb{R}$ is the **bias (offset/brightness)**. It translates the entire histogram uniformly up or down.

### 1.2 Defining the Statistical Targets
Given an image with $N$ pixels, we describe its distribution using two fundamental statistical moments:
1. **First Moment (Mean Brightness, $\mu$):**
   $$\mu(I') = \frac{1}{N} \sum_{x, y} I'(x, y)$$

2. **Second Central Moment (Contrast / Standard Deviation, $\sigma$):**
   $$\sigma(I') = \sqrt{\frac{1}{N} \sum_{x, y} \left( I'(x, y) - \mu(I') \right)^2}$$

Because the model is linear, note how $s$ and $b$ propagate mathematically:
$$\mu(I') = s \cdot \mu(I) + b$$
$$\sigma(I') = |s| \cdot \sigma(I)$$

### 1.3 Underdetermined vs. Well-Posed Systems
- **Why Brightness alone fails:** If our residual vector is only $\mathbf{r}(s, b) = [\mu(I') - \mu_{\text{target}}]$, we have 1 scalar equation and 2 unknowns ($s, b$). This is an **underdetermined system** with infinite solutions satisfying:
  $$b = \mu_{\text{target}} - s \cdot \mu(I)$$
  Setting $s = 0$ and $b = \mu_{\text{target}}$ yields an error of zero, but destroys all image details ($\sigma = 0$).
- **The Well-Posed Formulation:** Constraining both moments gives 2 equations for 2 unknowns:
  $$\mathbf{r}(s, b) = \begin{bmatrix} \mu(I') - \mu_{\text{target}} \\ \sigma(I') - \sigma_{\text{target}} \end{bmatrix}$$
  This isolates a unique, stable global minimum.

### 1.4 API Mechanics & Minimal Isolated Snippet
- `scipy.optimize.least_squares(fun, x0, args=...)`: Solves $\min \frac{1}{2} \|\mathbf{r}\|_2^2$.
- `np.clip(array, 0, 255)`: Clamps floating-point pixel values into the valid display dynamic range before casting to `uint8`.

```python
import numpy as np
from scipy.optimize import least_squares

def residual_brightness_contrast(params: np.ndarray, img: np.ndarray, target_mu: float, target_sigma: float) -> np.ndarray:
    s, b = params
    # 1. Apply affine model to image matrix
    transformed = s * img + b
    
    # 2. Compute current sample statistics
    current_mu = np.mean(transformed)
    current_sigma = np.std(transformed)
    
    # 3. Formulate residual vector r(θ)
    return np.array([current_mu - target_mu, current_sigma - target_sigma], dtype=np.float64)

# Isolated Demonstration:
dummy_img = np.random.uniform(50, 180, size=(200, 200)).astype(np.float64)
x0 = np.array([1.0, 0.0])  # Start with identity transform: s=1, b=0

res = least_squares(residual_brightness_contrast, x0, args=(dummy_img, 80.0, 30.0))
s_opt, b_opt = res.x
print(f"Result: s = {s_opt:.4f}, b = {b_opt:.4f}")
```

---

## 2. Exercises 2 & 3: Photometric Alignment in Mosaicking

### 2.1 The Overlap Domain ($\Omega$)
When two images are geometrically registered—a reference train image $I_t(x, y)$ and a warped query image $I_q(x, y)$—we define the active comparison mask $\Omega$ as the set of pixel coordinates where valid data exists in both images:

$$\Omega = \{ (x, y) \mid I_t(x, y) > 0 \;\land\; I_q(x, y) > 0 \}$$

Inside $\Omega$, each pixel coordinate provides a local measurement.

```
┌──────────────────────────────────────┐
│ Train Image It(x, y)                 │
│             ┌────────────────────────┼────────────┐
│             │  Active Overlap (Ω)    │            │
│             │  r_xy = (s*Iq + b) - It│ Warped Iq  │
│             └────────────────────────┼────────────┘
└──────────────────────────────────────┘
```

### 2.2 Pixel-Wise Similarity Residuals (Exercise 2)
The objective is to minimize point-to-point radiometric differences across all coordinates in $\Omega$:

$$r_{xy}(s, b) = \left( s \cdot I_q(x, y) + b \right) - I_t(x, y), \quad \forall (x, y) \in \Omega$$

#### Why Simultaneous Correction Degenerates (Step 2.1):
If we allow both images to vary ($s_t, b_t$ for $I_t$, and $s_q, b_q$ for $I_q$):
$$r_{xy} = (s_q I_q + b_q) - (s_t I_t + b_t)$$
The optimizer finds the trivial global minimum: $s_q = s_t = 0$ and $b_q = b_t = 0$. The residual becomes $0 - 0 = 0$, producing two pitch-black images.  
**Rule:** You must anchor the reference frame by holding $I_t$ fixed ($s_t = 1, b_t = 0$) and optimizing only $s_q, b_q$.

#### Vector Residual vs. Scalar Mean (Step 2.3):
- Returning a single scalar average error $\frac{1}{|\Omega|} \sum r_{xy}$ causes catastrophic cancellation: a positive error of $+50$ at one pixel cancels a negative error of $-50$ at another, reporting zero error even though the images look completely mismatched.
- Returning the full $M$-dimensional vector ($M = |\Omega|$) forces the sum of squares $\sum r_{xy}^2$ to drive every individual pixel difference toward zero.

### 2.3 Multi-Objective Balancing (Exercise 3)
In Exercise 3, we optimize both images ($s_t, b_t, s_q, b_q$), but prevent degeneration by anchoring the entire mosaic to a global target brightness ($\mu^*$) and target contrast ($\sigma^*$).

The combined residual vector contains:
1. **Pixel similarity residuals:** $M$ terms ($M \approx 10^5 \text{ to } 10^6$ pixels).
2. **Statistical target residuals:** 4 terms ($\mu_t - \mu^*$, $\sigma_t - \sigma^*$, $\mu_q - \mu^*$, $\sigma_q - \sigma^*$).

Because least squares minimizes the total sum of squares:
$$\Phi = \sum_{i=1}^M r_{\text{pixel}, i}^2 + \sum_{j=1}^4 (w \cdot r_{\text{stat}, j})^2$$

If $w = 1$, the $10^5$ pixel residuals completely dominate the 4 statistical residuals, making the statistical targets completely ignored.  
To equalize their gradients, we introduce a balancing weight:

$$w \approx \sqrt{\frac{M}{4}} = \sqrt{\frac{|\Omega|}{4}}$$

This scales the variance of the 4 statistical equations so they carry equal voting power in the Jacobian matrix.

### 2.4 API Mechanics & Minimal Isolated Snippet
```python
import numpy as np
from scipy.optimize import least_squares

def residual_mosaic(params: np.ndarray, q_vals: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    s, b = params
    # 1D vector of differences across all overlapping pixels
    return (s * q_vals + b) - t_vals

# Extracting overlap values via NumPy boolean masking
mask = (query_warped > 0) & (train_img > 0)
q_overlap = query_warped[mask].astype(np.float64)
t_overlap = train_img[mask].astype(np.float64)

res = least_squares(residual_mosaic, x0=[1.0, 0.0], args=(q_overlap, t_overlap))
s_opt, b_opt = res.x
```

---

## 3. Exercise 4: Geometric Alignment via SIFT & Least Squares Translation

### 3.1 Geometric Formulation (2D Translation)
Assuming the camera motion between $I_q$ and $I_t$ is purely planar translation with no rotation or scale change, a point $\mathbf{p}_q = [x_q, y_q]^T$ in the query image maps to $\mathbf{p}_t = [x_t, y_t]^T$ in the train image via translation vector $\mathbf{t} = [\Delta x, \Delta y]^T$:

$$\begin{bmatrix} x_t \\ y_t \end{bmatrix} = \begin{bmatrix} x_q \\ y_q \end{bmatrix} + \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}$$

### 3.2 Keypoint Detection & Feature Matching
1. **SIFT (`cv.SIFT_create`):** Identifies scale-space extrema (keypoints) and computes 128-dimensional gradient orientation histograms (descriptors).
2. **K-Nearest Neighbors (`knnMatch` with $k=2$):** Finds the two closest descriptor candidates in feature space.
3. **Lowe's Ratio Test:** Discards ambiguous matches by enforcing:
   $$\frac{\|\mathbf{d}_q - \mathbf{d}_{t,1}\|_2}{\|\mathbf{d}_q - \mathbf{d}_{t,2}\|_2} < \tau \quad (\text{typically } \tau = 0.75)$$

### 3.3 The Least Squares Residuals
For $P$ matched pairs $\{(\mathbf{p}_{q,i}, \mathbf{p}_{t,i})\}_{i=1}^P$, the 2D reprojection residual for each pair is:

$$\mathbf{r}_i(\Delta x, \Delta y) = \begin{bmatrix} (x_{q,i} + \Delta x) - x_{t,i} \\ (y_{q,i} + \Delta y) - y_{t,i} \end{bmatrix}$$

The total cost function is:
$$E(\Delta x, \Delta y) = \sum_{i=1}^P \left[ (x_{q,i} + \Delta x - x_{t,i})^2 + (y_{q,i} + \Delta y - y_{t,i})^2 \right]$$

### 3.4 Mathematical Proof: Why Least Squares Equals the Mean Difference
To find the minimum, we set the partial derivative with respect to $\Delta x$ to zero:

$$\frac{\partial E}{\partial \Delta x} = \sum_{i=1}^P 2(x_{q,i} + \Delta x - x_{t,i}) = 0$$

Expanding the summation:
$$2 \left( \sum_{i=1}^P x_{q,i} + P \cdot \Delta x - \sum_{i=1}^P x_{t,i} \right) = 0$$

$$P \cdot \Delta x = \sum_{i=1}^P (x_{t,i} - x_{q,i})$$

$$\Delta x^* = \frac{1}{P} \sum_{i=1}^P (x_{t,i} - x_{q,i})$$

This confirms analytically that **the optimal least-squares shift under Gaussian noise is exactly the arithmetic mean of all pairwise displacements**.

### 3.5 API Mechanics & Minimal Isolated Snippet
```python
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

# Feature Detection and Lowe's Matching
sift = cv.SIFT_create()
kp_q, des_q = sift.detectAndCompute(img_query, None)
kp_t, des_t = sift.detectAndCompute(img_train, None)

bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des_q, des_t, k=2)

pts_q, pts_t = [], []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        pts_q.append(kp_q[m.queryIdx].pt)
        pts_t.append(kp_t[m.trainIdx].pt)

pts_q = np.array(pts_q, dtype=np.float64)  # Shape: (P, 2)
pts_t = np.array(pts_t, dtype=np.float64)

# Residual formulation: returns 1D array of size 2*P
def residual_geom(delta: np.ndarray, p_q: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    diff = (p_q + delta) - p_t
    return diff.flatten()

# Optimization with bounding box constraints:
# bounds = ([min_dx, min_dy], [max_dx, max_dy])
res = least_squares(residual_geom, x0=[0.0, 0.0], args=(pts_q, pts_t), bounds=([-1000, -1000], [1000, 1000]))
dx_opt, dy_opt = res.x
```

---

## 4. Exercise 5: Outliers, RANSAC, and M-Estimators (Robust Optimization)

### 4.1 The Sensitivity of Ordinary Least Squares
In standard $L_2$ regression, the loss function is quadratic: $\rho(r) = \frac{1}{2} r^2$.
Its influence function (the derivative representing pull on parameters) is linear:
$$\psi(r) = \frac{d\rho}{dr} = r$$

If a false match creates an error of $r = 500 \text{ pixels}$, its pull is 500 times stronger than a 1-pixel error, severely corrupting the estimate $\Delta x^*, \Delta y^*$.

```
Quadratic Loss (L2):        ρ(r) = r²         ──► Large errors explode!
Soft L1 Loss (M-Estimator): ρ(r) = 2(√(1+r²) - 1) ──► Large errors grow linearly!
```

### 4.2 Robust M-Estimators (`loss='soft_l1'`)
SciPy's `least_squares` allows replacing the squared loss with a robust M-estimator.
The **Soft L1** loss is defined as:

$$\rho(z) = 2 \left( \sqrt{1 + z} - 1 \right), \quad \text{where } z = \left( \frac{r}{\text{f\_scale}} \right)^2$$

- When $r \ll \text{f\_scale}$ (inliers): $\sqrt{1 + z} \approx 1 + \frac{1}{2}z \implies \rho(z) \approx z$ (behaves quadratically, like $L_2$).
- When $r \gg \text{f\_scale}$ (outliers): $\rho(z) \approx 2\sqrt{z} \propto |r|$ (behaves linearly, bounding the influence of massive outliers).

### 4.3 RANSAC Consensus
`cv.findHomography` with `cv.RANSAC`:
1. Randomly draws minimal subsets of points.
2. Fits the transformation model.
3. Computes the geometric distance $d(\mathbf{p}_{t,i}, \mathbf{p}_{q,i} + \mathbf{t})$ for all points.
4. Points with $d < \text{threshold}$ are classified as inliers (`mask[i] = 1`). All others are flagged as outliers (`mask[i] = 0`).

### 4.4 API Mechanics & Minimal Isolated Snippet
```python
# 1. RANSAC Filtering
H, inlier_mask = cv.findHomography(pts_q, pts_t, cv.RANSAC, ransacReprojThreshold=5.0)
inlier_indices = np.where(inlier_mask.flatten() == 1)[0]
pts_q_inliers = pts_q[inlier_indices]
pts_t_inliers = pts_t[inlier_indices]

# 2. Robust Least Squares using Soft L1
res_robust = least_squares(
    residual_geom,
    x0=[0.0, 0.0],
    args=(pts_q, pts_t),
    loss='soft_l1',
    f_scale=2.0  # Threshold in pixels where linear attenuation begins
)
```

---

## 5. Exercise 6: Joint Geometric & Photometric Direct Alignment

### 5.1 Direct Formulation (Pixel-Level Tracking)
Instead of relying on sparse keypoints, direct methods align images using raw intensity values across all query coordinates $(x, y) \in \text{dom}(I_q)$:

$$e(x, y) = \underbrace{s \cdot I_q(x, y) + b}_{\text{Photometric adjustment}} - \underbrace{I_t(x + \Delta x, y + \Delta y)}_{\text{Spatially shifted reference}}$$

The joint parameter vector to estimate is:
$$\mathbf{\theta} = \begin{bmatrix} \Delta x \\ \Delta y \\ s \\ b \end{bmatrix} \in \mathbb{R}^4$$

### 5.2 Sub-Pixel Bilinear Interpolation via `cv.remap`
Because the optimizer steps continuously through real numbers $(\Delta x, \Delta y) \in \mathbb{R}^2$, the sample location $(x + \Delta x, y + \Delta y)$ is generally not an integer.
To obtain smooth, differentiable derivatives for gradient descent, we interpolate intensity values bilinearly using `cv.remap`:

$$I(x + \Delta x, y + \Delta y) \approx (1 - \alpha)(1 - \beta) I_{i,j} + \alpha(1 - \beta) I_{i+1,j} + (1 - \alpha)\beta I_{i,j+1} + \alpha\beta I_{i+1,j+1}$$

where $i = \lfloor x + \Delta x \rfloor, \alpha = (x + \Delta x) - i$, and $j = \lfloor y + \Delta y \rfloor, \beta = (y + \Delta y) - j$.

### 5.3 Non-Convex Optimization & Local Minima
Pixel-intensity objective landscapes are notoriously rugged:
- High-frequency textures (foliage, sharp edges) generate numerous local minima.
- Starting far away from the global optimum causes gradient descent to get trapped in an incorrect local valley.

```
Error Landscape:
     Local minimum (wrong match)
         ▼           Global Minimum (True alignment)
/\      /\                  ▼
  \    /  \      /\        / \
   \  /    \____/  \______/   \
    \/
```

#### Engineering Solution (Coarse-to-Fine):
1. **Gaussian Multi-Scale & Blurring:** Downsampling (e.g. scale $\times \frac{1}{8}$) and Gaussian smoothing (`cv.GaussianBlur`) flatten out local high-frequency dips, widening the basin of attraction around the true minimum.
2. **Template Matching Initialization:** `cv.matchTemplate` with `cv.TM_CCOEFF_NORMED` performs a fast exhaustive grid search, delivering an initial integer estimate $(\Delta x_0, \Delta y_0)$ close to the global bowl.
3. **Sub-Pixel Refinement:** Starting from this basin, `least_squares` smoothly refines the coordinates to fractional precision while simultaneously estimating $s$ and $b$.

### 5.4 API Mechanics & Minimal Isolated Snippet
```python
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

def warp_sample(img: np.ndarray, dx: float, dy: float, out_h: int, out_w: int) -> np.ndarray:
    # Build 2D coordinate meshgrids
    grid_x, grid_y = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    map_x = grid_x + np.float32(dx)
    map_y = grid_y + np.float32(dy)
    # Perform bilinear interpolation with zero border padding
    return cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)

def residual_direct(theta: np.ndarray, img_q: np.ndarray, img_t: np.ndarray) -> np.ndarray:
    dx, dy, s, b = theta
    h_q, w_q = img_q.shape[:2]
    
    # 1. Sample train image at shifted query locations
    t_sampled = warp_sample(img_t, dx, dy, h_q, w_q)
    
    # 2. Apply photometric correction to query image
    q_corrected = s * img_q + b
    
    # 3. Mask valid overlapping pixels (ignore black borders)
    valid = (t_sampled > 0)
    
    # Return 1D residual vector
    return (q_corrected[valid] - t_sampled[valid]).flatten()

# --- Workflow Execution ---
# 1. Downscale images (e.g., scale factor = 0.125)
scale = 0.125
q_small = cv.resize(img_query, (0, 0), fx=scale, fy=scale)
t_small = cv.resize(img_train, (0, 0), fx=scale, fy=scale)

# 2. Template Matching for Global Coarse Estimation
tm_result = cv.matchTemplate(t_small, q_small, cv.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv.minMaxLoc(tm_result)
init_dx, init_dy = max_loc

# 3. Fine Continuous Optimization
x0 = np.array([init_dx, init_dy, 1.0, 0.0], dtype=np.float64)
opt_res = least_squares(residual_direct, x0, args=(q_small, t_small))
dx_small, dy_small, s_opt, b_opt = opt_res.x

# Scale back translation to full-resolution image coordinate space
dx_full = dx_small / scale
dy_full = dy_small / scale
```
