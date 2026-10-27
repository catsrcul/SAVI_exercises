# SAVI - Part 5: Optimization and Model Fitting

## Overview

This module covers interactive data collection techniques and numerical optimization algorithms for curve fitting (model fitting):
1. **Interactive Data Acquisition (GUI Event Handling with Matplotlib)** and JSON persistence.
2. **Stochastic / Random Fitting (Random Line Fitting)** via Monte Carlo sampling and error metric evaluation.
3. **Nonlinear Least Squares Fitting** using the Levenberg-Marquardt / Trust Region Reflective algorithms via `scipy.optimize.least_squares`.
4. **Piecewise / Composed Model Fitting** combining distinct functional regimes ($x \le 0$ and $x > 0$).

---

## 1. Exercise 1: Interactive Point Generation (`generate_data`)

### Theoretical Background & Event Loop
Matplotlib's graphical user interface operates via an event loop driven by operating system interrupts (mouse and keyboard).
To record discrete user samples:
- Bind the click event `button_press_event` to a callback function.
- Convert the physical cursor coordinates into normalized data coordinates: $(x_{\text{data}}, y_{\text{data}})$.
- Terminate user input via the keyboard event `key_press_event` upon intercepting the `escape` key.

### Implementation Mechanics and Signatures
- Registering event listeners: `fig.canvas.mpl_connect(event_name: str, callback_fn: Callable)`.
- Exporting data to a dictionary: `{"xs": [...], "ys": [...]}`.
- Serialization via `json.dump(obj, fp, indent=4)`.

```python
import matplotlib.pyplot as plt
import json

xs, ys = [], []

def on_click(event):
    if event.inaxes is not None:
        xs.append(float(event.xdata))
        ys.append(float(event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        event.canvas.draw_idle()

def on_key(event):
    if event.key == 'escape':
        plt.close()

fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 5)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

with open("points.json", "w") as f:
    json.dump({'xs': xs, 'ys': ys}, f, indent=4)
```

---

## 2. Exercise 2: Random Line Fitting (Monte Carlo Search)

### Mathematical Formulation
Given a set of $N$ points $\{(x_i, y'_i)\}_{i=0}^{N-1}$ and an affine model:

$$y_i = m \cdot x_i + b$$

The objective is to determine parameters $\mathbf{p}^* = (m^*, b^*)$ that minimize cumulative error. To prevent positive and negative residuals from canceling each other out, either $L_1$ (MAE) or $L_2$ (MSE) norms are used:

$$J_{L1}(m, b) = \sum_{i=0}^{N-1} |(m \cdot x_i + b) - y'_i|$$

$$J_{L2}(m, b) = \sum_{i=0}^{N-1} ((m \cdot x_i + b) - y'_i)^2$$

### Numerical Mechanics
In a uniform random search over predefined parameter intervals $[m_{\min}, m_{\max}]$ and $[b_{\min}, b_{\max}]$:
- Draw candidate samples: `m = np.random.uniform(m_low, m_high)`.
- Compute the vector cost in NumPy: `err = np.sum(np.abs((m * x + b) - y))`.
- Track and keep the parameter pair with the lowest accumulated cost.

```python
import numpy as np

def evaluate_random_line(x_arr: np.ndarray, y_arr: np.ndarray, n_iter: int = 1000):
    best_error = float('inf')
    best_params = (0.0, 0.0)
    
    for _ in range(n_iter):
        m_cand = np.random.uniform(-3.0, 3.0)
        b_cand = np.random.uniform(-5.0, 5.0)
        
        y_pred = m_cand * x_arr + b_cand
        error = np.sum(np.abs(y_pred - y_arr))
        
        if error < best_error:
            best_error = error
            best_params = (m_cand, b_cand)
            
    return best_params, best_error
```

---

## 3. Exercise 3: Least Squares Fitting (`scipy.optimize.least_squares`)

### Optimizer Formulation
Unlike blind stochastic search, gradient-based numerical methods converge deterministically toward local/global minima.
The problem is formulated in vector form:

$$\min_{\mathbf{p}} \frac{1}{2} \|\mathbf{r}(\mathbf{p})\|_2^2 = \min_{\mathbf{p}} \frac{1}{2} \sum_{i=0}^{N-1} r_i(\mathbf{p})^2$$

Where $\mathbf{r}(\mathbf{p}) \in \mathbb{R}^N$ is the **residual vector**:

$$r_i(\mathbf{p}) = y_{\text{model}}(x_i; \mathbf{p}) - y'_i$$

### SciPy API Mechanics
- **Signature:** `scipy.optimize.least_squares(fun, x0, args=(), ...)`
- `fun`: A callable with signature `fun(params, *args) -> np.ndarray`. It must return a **1D residual vector of length $N$**, and **not** a scalar sum.
- `x0`: Array containing initial parameter estimates.
- The optimized parameter vector is accessed via the `.x` attribute of the returned `OptimizeResult` object.

```python
from scipy.optimize import least_squares
import numpy as np

def residuals(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m, b = params
    return (m * x + b) - y

# Running optimizer with x0 and arguments
initial_guess = [1.0, 0.0]
res = least_squares(residuals, x0=initial_guess, args=(x_data, y_data))
m_opt, b_opt = res.x
```

---

## 4. Exercise 4: Piecewise / Composed Model Fitting

### Mathematical Formulation
The model divides the domain of $x$ into two separate functional regimes:

$$y(x; \mathbf{p}) = \begin{cases} 
m \cdot x + b_1 & \text{if } x \le 0 \\
a \cdot x^2 + c \cdot x + b_2 & \text{if } x > 0 
\end{cases}$$

Where $\mathbf{p} = [m, b_1, a, c, b_2] \in \mathbb{R}^5$.

### Vectorization Mechanics (`np.where`)
NumPy evaluates piecewise branches without explicit Python loops using vectorized boolean masking:

```python
def predict_composed(x: np.ndarray, m: float, b1: float, a: float, c: float, b2: float) -> np.ndarray:
    return np.where(x <= 0, m * x + b1, a * (x ** 2) + c * x + b2)
```

The residual function computes the element-wise difference:

```python
def composed_residuals(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_pred = predict_composed(x, *params)
    return y_pred - y
```
