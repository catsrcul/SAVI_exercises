# SAVI - Part 6: Optimization in Computer Vision (Step-by-Step Learning Guide)

Welcome to Part 6! If you found the previous math-heavy documentation overwhelming, this guide is designed specifically for you.

Here, we will break down **why** we do each step, connect the concepts to real-world engineering intuition, explain what the code is actually doing under the hood, and walk you through how to construct your solution step-by-step.

---

## 0. The Big Picture: What is "Optimization" in Computer Vision?

In engineering (like mechanical systems or control theory), you often want to tune knobs until a machine runs smoothly or hits a specific target.

In computer vision, **optimization** does the exact same thing:
1. You have some **parameters** you want to find (e.g., how much to brighten an image, or how many pixels to slide an image to the left or right).
2. You define an **error (or residual)**: "How far is my current result from where I want it to be?"
3. You let an algorithm (`scipy.optimize.least_squares`) automatically test and adjust those parameters until the total error becomes as small as possible.

```
[Initial Guess for Parameters]
           │
           ▼
     Apply to Image
           │
           ▼
   Calculate Residuals: (Current - Desired)
           │
    Are residuals small? 
     ├── NO  ──► Optimizer adjusts parameters and repeats
     └── YES ──► Done! Optimal parameters found.
```

---

## 1. Exercise 1: Controlling Brightness & Contrast with a Linear Model

### What is the Goal?
You are given an image (`images/marvao/1.png`). You want to transform it so that it reaches an exact **target average brightness** and an exact **target contrast** (standard deviation).

### The Physical / Mathematical Intuition
Every grayscale image is just a 2D matrix of numbers between `0` (pitch black) and `255` (pure white).
To change how bright or contrasted an image is, we apply a linear equation to every single pixel $(x, y)$:

$$I'(x, y) = s \cdot I(x, y) + b$$

- **$s$ (Scale / Gain):** Acts like contrast. If $s > 1$, bright pixels get brighter and dark pixels get darker (expanding the range). If $s < 1$, everything shrinks closer together (reducing contrast).
- **$b$ (Bias / Offset):** Acts like overall brightness. It shifts every pixel up or down by a constant number.

### Why do we need both Brightness AND Contrast?
- **Mean brightness ($\mu$):** The average value of all pixels in the image.
- **Contrast ($\sigma$):** How spread out pixel values are from the average (the standard deviation).

> **Crucial Engineering Insight:**  
> If you only tell the optimizer "make the average brightness 80", you have **2 variables ($s$ and $b$)** but only **1 requirement**. There are infinite combinations that give an average of 80! (For instance, $s=0, b=80$ makes every pixel exactly 80, but contrast becomes 0!).  
> To get a unique, well-behaved solution, you must constrain both:
> 1. Error 1: $\mu_{\text{current}} - \mu_{\text{target}}$
> 2. Error 2: $\sigma_{\text{current}} - \sigma_{\text{target}}$

### How `least_squares` works here
You give `least_squares` a Python function that accepts `params = [s, b]`.
Inside that function, you:
1. Calculate the transformed image: `transformed = s * img + b`
2. Measure its current mean: `np.mean(transformed)`
3. Measure its current std: `np.std(transformed)`
4. Return the list of differences: `[mean - target_mean, std - target_std]`

The optimizer will automatically tweak $s$ and $b$ until both differences are 0.

### Self-Contained Learning Example
```python
import numpy as np
from scipy.optimize import least_squares

def cost_function(params, image_data, target_brightness, target_contrast):
    s, b = params
    # Apply s and b to the image array
    transformed = s * image_data + b
    
    # Calculate current properties
    current_mean = np.mean(transformed)
    current_std = np.std(transformed)
    
    # Return the residuals (differences to minimize)
    err_brightness = current_mean - target_brightness
    err_contrast = current_std - target_contrast
    return [err_brightness, err_contrast]

# Synthetic test
test_image = np.array([50.0, 100.0, 150.0, 200.0])
initial_guess = [1.0, 0.0]  # Start with no scale change (s=1) and no offset (b=0)

result = least_squares(cost_function, initial_guess, args=(test_image, 80.0, 30.0))
s_opt, b_opt = result.x
print(f"Optimal scale s: {s_opt:.3f}, bias b: {b_opt:.3f}")
```

---

## 2. Exercise 2 & 3: Making Two Images Match in Color (Photometric Stitching)

### What is the Goal?
In panorama stitching, you take two overlapping photos (`machu_pichu/1.png` and `query_warped.png`). Even if they align physically, one camera shot might be darker or have different lighting than the other. You want to adjust the colors so the seam disappears.

### The Intuition of the Overlap Zone
The image `query_warped.png` is already aligned with `1.png`, but has black borders where there is no data.
We only want to compare pixels where **both** images actually exist:

```
┌───────────────────────────┐
│ Train Image (1.png)       │
│        ┌──────────────────┼────────┐
│        │  OVERLAP ZONE    │        │
│        │ (Compare here!)  │ Query  │
│        └──────────────────┼────────┘
└───────────────────────────┘
```

In NumPy, you isolate this zone using a mask:
```python
# Where query has content (not black/zero):
mask = (query_warped > 0)
train_pixels = train_img[mask]
query_pixels = query_warped[mask]
```

### Why does optimizing BOTH images fail? (Exercise 2.1)
If you ask the optimizer to tune $s_1, b_1$ for image 1 and $s_2, b_2$ for image 2 to make them as close as possible:
$$\text{Difference} = (s_1 \cdot I_1 + b_1) - (s_2 \cdot I_2 + b_2)$$
The optimizer will realize that setting $s_1 = s_2 = 0$ and $b_1 = b_2 = 0$ results in:
$$0 - 0 = 0 \quad (\text{Zero error!})$$
Both images become pitch black! To avoid this, one image **must act as an immovable reference** (keep image 1 fixed, and only tune $s, b$ for image 2).

### Multi-Objective Balancing (Exercise 3)
In Exercise 3, you want both images to match, **and** you want the final picture to have a specific overall brightness and contrast.
Now you have:
- Over $100,000$ pixel error values (one for each pixel in the overlap).
- Only $2$ or $4$ brightness/contrast error values.

Because `least_squares` minimizes the sum of all squared errors, 100,000 pixel errors will completely overpower the 4 brightness errors! To give them fair voting power, you must multiply the brightness/contrast errors by a weight factor:
$$w \approx \sqrt{\frac{\text{number of overlap pixels}}{\text{number of global stats}}}$$

---

## 3. Exercise 4: Finding the Position (Geometric Alignment via SIFT Features)

### What is the Goal?
Instead of assuming the query image is already placed in the right spot, you must find the horizontal and vertical shift $(\Delta x, \Delta y)$ that moves the query image directly on top of the train image.

### How SIFT Works Conceptually
1. **Detect Keypoints:** SIFT finds unique, high-contrast corner-like points in both images (e.g., corners of a mountain or building).
2. **Compute Descriptors:** Around each point, it creates a unique numerical fingerprint (a vector of 128 numbers).
3. **Match:** It finds pairs of points across both images that have nearly identical fingerprints.

### Turning Matches into a Translation Optimization
Let a matched point in the query image be $(x_q, y_q)$ and its partner in the train image be $(x_t, y_t)$.
If we shift the query image by $(\Delta x, \Delta y)$, the new position is $(x_q + \Delta x, y_q + \Delta y)$.

The error between where it lands and where it should be is:
$$e_x = (x_q + \Delta x) - x_t$$
$$e_y = (y_q + \Delta y) - y_t$$

`least_squares` tests different $(\Delta x, \Delta y)$ values until these distances are minimized across all matched pairs.

### Why is this the same as the Average Difference?
If you take the derivative of the squared error sum with respect to $\Delta x$ and set it to 0, you get:
$$\Delta x^* = \frac{1}{N} \sum_{i=1}^N (x_{t,i} - x_{q,i})$$
The optimal least-squares translation is simply the **average shift** of all matched points!

### Isolated SIFT and Matching Snippet
```python
import cv2 as cv
import numpy as np

# 1. Initialize SIFT detector
sift = cv.SIFT_create()

# 2. Find keypoints and descriptors
kp_train, desc_train = sift.detectAndCompute(img_train, None)
kp_query, desc_query = sift.detectAndCompute(img_query, None)

# 3. Match descriptors using Brute-Force Matcher
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(desc_query, desc_train, k=2)

# 4. Apply Lowe's Ratio Test (keeps only confident, distinctive matches)
good_query_pts = []
good_train_pts = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_query_pts.append(kp_query[m.queryIdx].pt)
        good_train_pts.append(kp_train[m.trainIdx].pt)

pts_q = np.array(good_query_pts)  # Shape (N, 2): column 0 is x, column 1 is y
pts_t = np.array(good_train_pts)
```

---

## 4. Exercise 5: Dealing with Bad Matches (Outliers & RANSAC)

### The Problem: Wrong Matches
SIFT is good, but not perfect. Occasionally, a rock on the left side of the mountain looks just like a rock on the right side. This produces an **outlier** (a match with a completely wrong vector line).

Because standard least squares squares all errors ($e^2$), a single bad match with an error of $500$ pixels contributes $500^2 = 250,000$ to the penalty! This single bad point will drag the entire image shift in the wrong direction.

```
Good Match 1:   ──────►  (shift: +120 px)
Good Match 2:   ──────►  (shift: +122 px)
Good Match 3:   ──────►  (shift: +119 px)
Bad Match:      ◄───────────────  (shift: -600 px! Ruins the average)
```

### The Solutions:
1. **RANSAC (`cv.findHomography` with `cv.RANSAC`):**
   RANSAC randomly picks a few points, calculates the shift, checks how many other points agree (inliers), and throws away the ones that disagree (outliers).
2. **Robust Loss Functions (`loss='soft_l1'` in `least_squares`):**
   Instead of squaring large errors, `soft_l1` treats large errors almost linearly, preventing a single runaway point from dominating the optimization.

### Minimal Code Example
```python
from scipy.optimize import least_squares

def translation_residuals(delta, pts_q, pts_t):
    dx, dy = delta
    shifted_q = pts_q + np.array([dx, dy])
    return (shifted_q - pts_t).flatten()

# Standard least squares (vulnerable to outliers)
res_standard = least_squares(translation_residuals, [0.0, 0.0], args=(pts_q, pts_t))

# Robust least squares (ignores outlier pull)
res_robust = least_squares(translation_residuals, [0.0, 0.0], args=(pts_q, pts_t), loss='soft_l1')
```

---

## 5. Exercise 6: Direct Alignment (Geometry + Color in One Step)

### What is the Goal?
Here, we throw away SIFT keypoints completely. We optimize all 4 parameters simultaneously:
- **$\Delta x, \Delta y$**: Where the query image belongs inside the train image.
- **$s, b$**: The color correction factors.

For every single pixel in the query image:
$$\text{Error}(x, y) = \underbrace{(s \cdot I_q(x, y) + b)}_{\text{Color corrected query}} - \underbrace{I_t(x + \Delta x, y + \Delta y)}_{\text{Train image at shifted location}}$$

### Why do we need `cv.remap`?
When $\Delta x$ is a fractional number like $15.42$ pixels, you cannot simply look up index `[y + 10.1, x + 15.42]` in a matrix because array indices must be integers.
`cv.remap` uses **bilinear interpolation** to blend surrounding pixels smoothly, providing accurate sub-pixel values.

### Why do we need Downscaling and Template Matching?
Pixel-level optimization has a notorious problem: **Local Minima**.
If you start too far away from the correct alignment, the optimizer will see mountains lining up with clouds or trees lining up with shadows and get permanently stuck.

To succeed:
1. **Downscale and Blur:** Shrink the images (e.g., to 1/8 size) and blur them. Blurring smooths out sharp rocks and leaves broad light/dark shapes that guide the optimizer from far away.
2. **Template Matching First:** Use `cv.matchTemplate` to find roughly where the query image sits (within a few pixels). Use that rough coordinate as the starting guess `x0` for `least_squares`.

### Isolated Direct Sampling Snippet
```python
import cv2 as cv
import numpy as np

def sample_translated_image(image, dx, dy, target_shape):
    h, w = target_shape[:2]
    # Create coordinate grid
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    
    # Add translation offsets
    map_x = grid_x + np.float32(dx)
    map_y = grid_y + np.float32(dy)
    
    # Sub-pixel sampling with bilinear interpolation
    sampled = cv.remap(image, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    return sampled
```

---

## 6. Practical Roadmap: How to Approach Writing the Code

When you sit down to implement Part 6, take it one step at a time:

1. **Start with Exercise 1:**
   - Load the image in grayscale, convert to `float64`.
   - Write a simple residual function calculating `[mean - target_mean, std - target_std]`.
   - Run `least_squares` and display the resulting image with `cv.imshow` (remember to use `np.clip(transformed, 0, 255).astype(np.uint8)`).
2. **Move to Exercise 2 & 3:**
   - Load `1.png` and `query_warped.png`.
   - Create a boolean mask of non-zero pixels.
   - Extract the 1D pixel arrays and optimize $s$ and $b$.
3. **Move to Exercise 4 & 5:**
   - Set up SIFT detection and BFMatcher.
   - Extract $(x, y)$ coordinates from keypoints.
   - Run least squares on translation and compare with RANSAC and `soft_l1`.
4. **Finish with Exercise 6:**
   - Downscale both images with `cv.resize`.
   - Find the initial $(\Delta x_0, \Delta y_0)$ with `cv.matchTemplate`.
   - Run joint optimization for $[\Delta x, \Delta y, s, b]$.
