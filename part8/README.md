# SAVI - Part 8: 3D Point Cloud Registration, Reference Frames & Object Bounding Boxes

https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html

Welcome to Part 8! In Part 7, we processed a single static point cloud. In real-world robotics and 3D perception, a single sensor scan only captures one side of a room or object (due to self-occlusion and field-of-view limits). To reconstruct a complete 3D model, we must capture multiple views and mathematically align them into a common coordinate frame—a process called **Point Cloud Registration**.

Once a complete 3D scene is registered, we can establish object-centric reference frames (such as a tabletop coordinate system) to segment individual objects and fit 3D **Bounding Boxes** around them.

---

## 0. The Big Picture: 3D Rigid Transformations & Error Metrics

### 0.1 Homogeneous Rigid Transformations in $\mathbf{SE}(3)$
To move a 3D point cloud (`part_2`, the **source** $\mathcal{P}_s$) into the coordinate system of another point cloud (`part_1`, the **target** $\mathcal{P}_t$), we apply a rigid-body transformation consisting of a 3D rotation matrix $\mathbf{R} \in \mathbf{SO}(3)$ ($3 \times 3$, where $\mathbf{R}^T \mathbf{R} = \mathbf{I}$ and $\det(\mathbf{R}) = +1$) and a 3D translation vector $\mathbf{t} \in \mathbb{R}^3$:

$$\mathbf{p}_t = \mathbf{R} \mathbf{p}_s + \mathbf{t}$$

Using **homogeneous coordinates** (appending a $1$ as the 4th coordinate), rotation and translation are combined into a single $4 \times 4$ matrix $\mathbf{T} \in \mathbf{SE}(3)$:

$$\begin{bmatrix} x_t \\ y_t \\ z_t \\ 1 \end{bmatrix} = \underbrace{\begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}}_{\mathbf{T}_{\text{target} \leftarrow \text{source}}} \begin{bmatrix} x_s \\ y_s \\ z_s \\ 1 \end{bmatrix}$$

### 0.2 Evaluating Registration Accuracy Against Ground Truth (`T_true.json`)
Suppose the ground-truth transformation is $\mathbf{T}_{\text{true}}$ and our algorithm estimates $\mathbf{T}$. How far off is our estimate?

In matrix algebra, we compute the **relative error transformation matrix** $\mathbf{E} \in \mathbf{SE}(3)$:

$$\mathbf{E} = \mathbf{T}_{\text{true}}^{-1} \mathbf{T} = \begin{bmatrix} \mathbf{R}_E & \mathbf{t}_E \\ \mathbf{0}^T & 1 \end{bmatrix}$$

> **Physical Intuition:** If you apply $\mathbf{T}$ to go from `part_2` to `part_1`, and then apply $\mathbf{T}_{\text{true}}^{-1}$ to go back from `part_1` to `part_2`, you should end up exactly where you started! If $\mathbf{T} = \mathbf{T}_{\text{true}}$, then $\mathbf{E} = \mathbf{I}_4$ (zero translation, zero rotation). Any residual motion inside $\mathbf{E}$ is your estimation error.

From $\mathbf{E}$, we extract two scalar engineering metrics:

1. **Translation Error ($e_t$, in millimeters):**
   The Euclidean norm (magnitude) of the residual translation vector $\mathbf{t}_E = \mathbf{E}_{0:3, 3}$:
   $$e_t = \|\mathbf{t}_E\|_2 = \sqrt{t_{E,x}^2 + t_{E,y}^2 + t_{E,z}^2} \quad [\text{m}] \xrightarrow{\times 1000} [\text{mm}]$$

2. **Rotation Error ($e_r$, in degrees):**
   By **Euler's Rotation Theorem**, any 3D rotation matrix $\mathbf{R}_E$ is equivalent to rotating by a single angle $\theta$ around some unit axis $\mathbf{u}$. A fundamental property of linear algebra is that the **trace** of a matrix (the sum of its diagonal elements) is invariant under change of basis, and for any 3D rotation matrix:
   $$\mathrm{tr}(\mathbf{R}_E) = r_{11} + r_{22} + r_{33} = 1 + 2\cos(\theta)$$
   Solving for the rotation angle $\theta$:
   $$\theta = \arccos\left(\frac{\mathrm{tr}(\mathbf{R}_E) - 1}{2}\right) = \arccos\left(\frac{r_{11} + r_{22} + r_{33} - 1}{2}\right)$$
   - When $\mathbf{R}_E = \mathbf{I}_3$, $\mathrm{tr}(\mathbf{I}_3) = 1 + 1 + 1 = 3 \implies \arccos\left(\frac{3-1}{2}\right) = \arccos(1) = 0^\circ$.
   - **Numerical Safety:** Floating-point inaccuracies can occasionally make $\frac{\mathrm{tr}(\mathbf{R}_E) - 1}{2} = 1.0000000002$, which causes `np.arccos` to return `NaN`. Always clamp the argument to $[-1, 1]$ using `np.clip` before calling `np.arccos`.

### 0.3 Minimal Isolated Snippet: Computing Transformation Error
```python
import json
import numpy as np
from scipy.spatial.transform import Rotation

def compute_registration_error(T_est: np.ndarray, T_true: np.ndarray) -> tuple[float, float]:
    # 1. Compute error matrix E = inv(T_true) @ T_est
    E = np.linalg.inv(T_true) @ T_est
    R_E = E[:3, :3]
    t_E = E[:3, 3]
    
    # 2. Translation error in mm (assuming input coordinates are in meters)
    trans_err_mm = float(np.linalg.norm(t_E) * 1000.0)
    
    # 3. Rotation error in degrees via matrix trace
    cos_theta = np.clip((np.trace(R_E) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(cos_theta)))
    
    # Optional verification via SciPy Rotation:
    # rot_err_scipy = np.degrees(Rotation.from_matrix(R_E).magnitude())
    return trans_err_mm, rot_err_deg

# Load ground-truth transformation from JSON
with open("icp_point_clouds/T_true.json", "r") as f:
    data = json.load(f)
T_true = np.array(data["T_part_1_part_2"], dtype=np.float64)
```

---

## 1. Exercise 1: Manual & ICP Registration in CloudCompare

### 1.1 Intuition: Why Coarse-to-Fine Registration?
When `part_1.ply` and `part_2.ply` are first loaded, they are separated by a large rotation and translation (several meters apart and rotated nearly $90^\circ$).
- **Coarse Registration (Point-Pair Picking):** If we manually identify at least $N \ge 3$ non-collinear corresponding 3D landmarks $\left\{(\mathbf{p}_{s,i}, \mathbf{p}_{t,i})\right\}_{i=1}^N$ between both clouds, we can solve for the optimal rigid transformation in closed form using Singular Value Decomposition (**Kabsch-Umeyama algorithm**). This places `part_2` right on top of `part_1`.
- **Fine Registration (ICP):** Manual clicking is only accurate to a few centimeters. Running Iterative Closest Point (ICP) afterward uses tens of thousands of points to lock the surfaces together down to millimeter precision.

### 1.2 Workflow in CloudCompare
1. Open `part_1.ply` (reference/target) and `part_2.ply` (aligned/source).
2. Select both clouds and open **Tools > Registration > Align (point pairs picking)**:
   - Set `part_1` as **Reference** and `part_2` as **Aligned**.
   - Click 3 or 4 distinct physical corners (e.g., corners of the table, distinctive objects, or chair backs) in the exact same order on both clouds.
   - Click **Align** and confirm.
3. With the roughly aligned cloud selected alongside `part_1`, open **Tools > Registration > Fine registration (ICP)** and run it.
4. Copy the final $4 \times 4$ transformation matrix printed in the CloudCompare console.

> **Important Note on Total Transformation:** If you perform two sequential transformations in CloudCompare—first $\mathbf{T}_{\text{coarse}}$ from point picking, and then $\mathbf{T}_{\text{icp}}$ on the already-moved cloud—the total transformation from the original `part_2` to `part_1` is the matrix product:
> $$\mathbf{T}_{\text{total}} = \mathbf{T}_{\text{icp}} \cdot \mathbf{T}_{\text{coarse}}$$

---

## 2. Exercise 2: Iterative Closest Point (ICP) in Open3D

Reference: [Open3D ICP Registration Tutorial](https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html)

### 2.1 How Does ICP Work Mathematically?
If we **knew** which point in `part_2` corresponded to which point in `part_1`, finding $\mathbf{T}$ would take a single step. Conversely, if we **knew** the true transformation $\mathbf{T}$, finding correspondences would just mean picking the nearest neighbor.

**Iterative Closest Point (ICP)** resolves this chicken-and-egg dilemma by alternating between two steps at every iteration $k$, starting from an initial guess $\mathbf{T}^{(0)}$:

```
       [Initial Guess T⁽⁰⁾]
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: Data Association (Nearest Neighbor Search)       │
│ For each transformed source point T⁽ᵏ⁾ p_i, find the     │
│ closest target point q_j within distance threshold d_max │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Transformation Update (Minimize Objective E(T))  │
│ Compute T⁽ᵏ⁺¹⁾ that minimizes distance between all       │
│ matched inlier pairs {(p_i, q_j)}                        │
└───────────────────────────┬──────────────────────────────┘
                            │
                  Converged?├─── NO ──► Repeat Step 1 with T⁽ᵏ⁺¹⁾
                            ▼ YES
                    [Return Final T*]
```

### 2.2 Point-to-Point vs. Point-to-Plane ICP

Open3D provides two main objective functions for Step 2:

#### 1. Point-to-Point ICP (`TransformationEstimationPointToPoint`)
Minimizes the sum of squared Euclidean distances directly between matched points $\mathbf{p}_i$ (in source) and $\mathbf{q}_i$ (in target):

$$E_{\text{p2p}}(\mathbf{R}, \mathbf{t}) = \sum_{(\mathbf{p}_i, \mathbf{q}_i) \in \mathcal{K}} \left\| \left(\mathbf{R} \mathbf{p}_i + \mathbf{t}\right) - \mathbf{q}_i \right\|_2^2$$

- **Limitation:** Points in a LiDAR/RGB-D scan are discrete samples of continuous surfaces. Two scans never sample the exact same 3D points on a flat wall. Pulling discrete points directly toward each other creates "friction" that resists sliding along flat surfaces.

#### 2. Point-to-Plane ICP (`TransformationEstimationPointToPlane`)
Instead of pulling $\mathbf{p}_i$ toward the discrete point $\mathbf{q}_i$, it projects the residual vector onto the **unit surface normal** $\mathbf{n}_i$ of the target point $\mathbf{q}_i$, minimizing distance to the **tangent plane**:

$$E_{\text{p2l}}(\mathbf{R}, \mathbf{t}) = \sum_{(\mathbf{p}_i, \mathbf{q}_i) \in \mathcal{K}} \left( \left[ \left(\mathbf{R} \mathbf{p}_i + \mathbf{t}\right) - \mathbf{q}_i \right] \cdot \mathbf{n}_i \right)^2$$

```
Point-to-Point:                       Point-to-Plane:
      ● p'_i (Transformed Source)           ● p'_i (Transformed Source)
     /                                      │
    /  ||(R p_i + t) - q_i||                │  ((R p_i + t) - q_i) · n_i
   ▼                                        ▼  (Allows free sliding along plane!)
──●─────────────── Target Surface     ──────┴────────●────── Target Surface (normal n_i ▲)
  q_i                                                q_i
```

- **Why Point-to-Plane is Superior:** Flat surfaces (walls, floors, tabletops) can slide freely along their tangent planes without penalty while snapping tightly along their normals. This yields **much faster convergence** (quadratic vs. linear behavior) and higher accuracy.

### 2.3 Role of `threshold` ($d_{\max}$), Initial Guess $\mathbf{T}^{(0)}$, and ICP Metrics
- **Why Initial Guess Matters (Local Minima):** Step 1 greedily pairs each point with whatever target point happens to be closest *right now*. If $\mathbf{T}^{(0)} = \mathbf{I}_4$ and the clouds are far apart, ICP pairs points from the floor of `part_2` with chairs of `part_1`, converging into a completely wrong **local minimum**.
- **Why Threshold ($d_{\max}$) Matters:**
  - Only pairs with $\|\mathbf{T}\mathbf{p}_i - \mathbf{q}_i\|_2 < d_{\max}$ are included in $\mathcal{K}$.
  - A **large threshold** (e.g., $0.2\text{ m}$) has a wider basin of attraction (captures correspondences from further away) but includes many false matches at the boundaries of partial overlap, biasing the final pose.
  - A **small threshold** (e.g., $0.02\text{ m}$) rejects outliers and achieves high accuracy, *provided* the clouds are already within $2\text{ cm}$ of each other.
  - **Multi-Scale Strategy:** Run ICP first with a larger threshold ($0.2\text{ m}$) from the rough initial guess to bring the clouds together, then pass that result as the initialization to a second ICP stage with a tight threshold ($0.02\text{ m}$).
- **Understanding Open3D's Output Metrics (`fitness` and `inlier_rmse`):**
  - **`fitness`** $\in [0, 1]$: Fraction of source points that found a target neighbor within $d_{\max}$:
    $$\text{fitness} = \frac{|\mathcal{K}|}{|\mathcal{P}_s|}$$
  - **`inlier_rmse`**: Root Mean Squared Error of all inlier correspondences in $\mathcal{K}$:
    $$\text{inlier\_rmse} = \sqrt{\frac{1}{|\mathcal{K}|} \sum_{(\mathbf{p}_i, \mathbf{q}_i) \in \mathcal{K}} d_i^2}$$
  - **Crucial Trap:** Are `fitness` and `inlier_rmse` alone enough to guarantee a good alignment? **No!**
    1. If $d_{\max}$ is tiny ($0.02\text{ m}$) and the clouds barely touch at a wrong angle, the few points that happen to intersect within $2\text{ cm}$ will have a tiny `inlier_rmse` ($\le 0.01\text{ m}$), even though the global registration is completely wrong!
    2. Conversely, in a room with flat walls and floors, locking two walls onto the wrong flat surfaces can yield high `fitness` and low `inlier_rmse` while objects are completely misaligned.

### 2.4 API Mechanics & Minimal Isolated Snippet
```python
import copy
import open3d as o3d
import numpy as np

def draw_registration_result(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             transformation: np.ndarray):
    # Copy clouds so we never mutate original coordinates in place
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1.0, 0.706, 0.0])  # Yellow (source: part_2)
    target_temp.paint_uniform_color([0.0, 0.651, 0.929]) # Cyan   (target: part_1)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# 1. Load clouds (target = part_1, source = part_2)
target = o3d.io.read_point_cloud("icp_point_clouds/part_1.ply")
source = o3d.io.read_point_cloud("icp_point_clouds/part_2.ply")

# 2. Estimate normals on target (required for Point-to-Plane ICP)
target.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
)

# 3. Point-to-Point ICP
T_init = np.eye(4)
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target,
    max_correspondence_distance=0.05,
    init=T_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print(f"P2P Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")

# 4. Point-to-Plane ICP
reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target,
    max_correspondence_distance=0.05,
    init=T_init,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
)
print(f"P2Plane Fitness: {reg_p2l.fitness:.4f}, RMSE: {reg_p2l.inlier_rmse:.4f}")
```

---

## 3. Exercise 3: Automatic Global Registration (FPFH + RANSAC)

Reference: [Open3D Global Registration Tutorial](https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html)

### 3.1 Theoretical & Mathematical Background
How do we align two point clouds automatically without any manual initial guess?
Just as we used 2D **SIFT descriptors + RANSAC** in Part 4 & Part 6 to align images, in 3D we use **FPFH descriptors + 3D RANSAC**:

1. **Voxel Downsampling & Normal Estimation:**
   Downsample both clouds with a voxel size $v_s$ (e.g., $v_s = 0.05\text{ m}$) and compute surface normals using a radius $r_{\text{normal}} = 2 v_s$.
2. **FPFH (Fast Point Feature Histograms):**
   For every point $\mathbf{p}$ and its neighbors within radius $r_{\text{feature}} = 5 v_s$, FPFH measures the relative angular variations $(\alpha, \phi, \theta)$ between pairs of surface normals and bins them into a **33-dimensional descriptor vector** $\mathbf{f}(\mathbf{p}) \in \mathbb{R}^{33}$.
   - **Key Property:** $\mathbf{f}(\mathbf{p})$ is **pose-invariant**—rotating or translating the point cloud does not change its FPFH histogram! A chair corner in `part_2` has the same 33D descriptor as the same chair corner in `part_1`.
3. **RANSAC with Geometric Pruning Checkers:**
   In each RANSAC iteration:
   - Sample $n = 3$ points from the downsampled source cloud and query their nearest neighbors in 33D FPFH feature space in the target cloud.
   - **Fast Pruning Before SVD (`CorrespondenceCheckerBasedOnEdgeLength`):** Before even computing a 3D transformation, check if the triangle formed by the 3 source points has the same side lengths as the triangle formed by the 3 matched target points (rigid transformations preserve edge lengths: $\|\mathbf{p}_i - \mathbf{p}_j\| \approx \|\mathbf{q}_i - \mathbf{q}_j\|$). This rejects $95\%+$ of bad matches immediately!
   - **Distance Checker (`CorrespondenceCheckerBasedOnDistance`):** Fit $\mathbf{T}$ to the 3 points and verify how many aligned points fall within distance threshold $d_{\text{ransac}} = 1.5 v_s$.
4. **Local ICP Refinement:**
   Global registration operates on coarse downsampled voxels ($v_s = 0.05\text{ m}$), so its output $\mathbf{T}_{\text{ransac}}$ is only accurate to a few centimeters. We pass $\mathbf{T}_{\text{ransac}}$ as `init` to Point-to-Plane ICP on the full-resolution clouds to obtain the final tight alignment.

### 3.2 API Mechanics & Minimal Isolated Snippet
```python
import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    # 1. Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 2. Estimate normals (radius = 2 * voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    
    # 3. Compute 33D FPFH descriptors (radius = 5 * voxel_size)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size: float):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result
```

---

## 4. Exercise 4: Tabletop Object Segmentation & Custom Reference Frames

In `scene_point_cloud/scene.ply`, the entire room is already registered. Our goal is to isolate the individual objects resting on the table and transform them into a **Table Reference Frame**.

### 4.1 Step 1: Ground Plane Detection & Signed Height Above Floor
A plane detected by RANSAC (`pcd.segment_plane`) has equation:
$$a x + b y + c z + d = 0, \quad \text{with unit normal } \mathbf{n} = [a, b, c]^T \quad (\|\mathbf{n}\|_2 = 1)$$

For any 3D point $\mathbf{p} = [x, y, z]^T$, its **signed orthogonal distance (height)** relative to the plane is:
$$h(\mathbf{p}) = a x + b y + c z + d = \mathbf{n} \cdot \mathbf{p} + d$$

> **Crucial Sign Check:** RANSAC does not know which way is "up"—it might return $(a, b, c, d)$ or $(-a, -b, -c, -d)$ (both represent the exact same geometric plane!).
> If the room lies on the negative side of $\mathbf{n}$, all heights $h(\mathbf{p})$ above the floor will be negative.
> **How to fix:** Evaluate the mean height of the remaining cloud (or check alignment with the camera/room vertical). If the majority of room points have $h(\mathbf{p}) < 0$, negate all 4 coefficients:
> $$[a, b, c, d] \leftarrow -[a, b, c, d]$$
> Now $h(\mathbf{p}) > 0$ is guaranteed to mean "above the floor".

### 4.2 Step 2: Isolating the Tabletop Plane
Why didn't we just run RANSAC on the whole room to find the table directly?
Because the **floor** (and walls) have far more points than the table! RANSAC always finds the plane with the largest number of inliers.
1. Filter points by height above the floor: keep only points with $0.5\text{ m} \le h_{\text{floor}}(\mathbf{p}) \le 1.0\text{ m}$.
2. Run `segment_plane` on this height slice to find the horizontal plane at table height.
3. Because chairs or shelves at the same height ($0.5\text{–}1.0\text{ m}$) might also fall on this infinite mathematical plane, run `cluster_dbscan` on the plane inliers and keep only the **largest cluster**—which is the physical tabletop $\mathcal{P}_{\text{table}}$.

### 4.3 Step 3: Constructing the Table Reference Frame ($\mathbf{T}_{\text{scene} \leftarrow \text{table}}$)
We want a coordinate frame attached to the table where:
- **Origin ($\mathbf{o}_{\text{table}}$):** The 3D centroid of the tabletop points:
  $$\mathbf{o}_{\text{table}} = \frac{1}{|\mathcal{P}_{\text{table}}|} \sum_{\mathbf{p} \in \mathcal{P}_{\text{table}}} \mathbf{p}$$
- **$Z$-axis ($\hat{\mathbf{z}}_{\text{table}}$):** The unit normal of the table plane, pointing **upward** (away from the floor, i.e., $\hat{\mathbf{z}}_{\text{table}} \cdot \mathbf{n}_{\text{floor}} > 0$).
- **$X$ and $Y$ axes ($\hat{\mathbf{x}}_{\text{table}}, \hat{\mathbf{y}}_{\text{table}}$):** Two mutually orthogonal unit vectors lying inside the table plane ($\hat{\mathbf{x}} \perp \hat{\mathbf{z}}$ and $\hat{\mathbf{y}} = \hat{\mathbf{z}} \times \hat{\mathbf{x}}$).

#### How to Construct an Orthonormal Basis from a Single Normal Vector $\hat{\mathbf{z}}$ (Gram-Schmidt / Cross-Product):
Given unit vector $\hat{\mathbf{z}}$:
1. Pick any helper vector $\mathbf{a}$ not parallel to $\hat{\mathbf{z}}$ (for example, $\mathbf{a} = [1, 0, 0]^T$ if $|z_x| < 0.9$, else $[0, 1, 0]^T$).
2. Compute $\hat{\mathbf{x}}$ perpendicular to $\hat{\mathbf{z}}$ via the cross product and normalize:
   $$\hat{\mathbf{x}} = \frac{\mathbf{a} \times \hat{\mathbf{z}}}{\|\mathbf{a} \times \hat{\mathbf{z}}\|_2}$$
3. Complete the right-handed triad via:
   $$\hat{\mathbf{y}} = \hat{\mathbf{z}} \times \hat{\mathbf{x}}$$
   *(Alternatively, you can obtain $\hat{\mathbf{x}}, \hat{\mathbf{y}}$ from the two largest eigenvectors of the tabletop points via PCA, ensuring $\det([\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{z}}]) = +1$).*

The pose of the table in the scene frame is then:
$$\mathbf{T}_{\text{scene} \leftarrow \text{table}} = \begin{bmatrix} \vert & \vert & \vert & \vert \\ \hat{\mathbf{x}}_{\text{table}} & \hat{\mathbf{y}}_{\text{table}} & \hat{\mathbf{z}}_{\text{table}} & \mathbf{o}_{\text{table}} \\ \vert & \vert & \vert & \vert \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

To transform all scene points **from the scene frame into the table frame**, we apply the **inverse** transformation:
$$\mathbf{p}_{\text{table}} = \mathbf{T}_{\text{table} \leftarrow \text{scene}} \, \mathbf{p}_{\text{scene}} = \left(\mathbf{T}_{\text{scene} \leftarrow \text{table}}\right)^{-1} \mathbf{p}_{\text{scene}}$$

### 4.4 Step 4: Cropping to the Tabletop Volume
Once transformed into the table frame:
- Every point on the tabletop surface has $z \approx 0$.
- Every point's $z$-coordinate is now literally its **height above the table**!
- Let $[x_{\min}, x_{\max}]$ and $[y_{\min}, y_{\max}]$ be the coordinate bounds of the tabletop points $\mathcal{P}_{\text{table}}$ in the table frame.
- We keep only points satisfying:
  $$x_{\min} \le x \le x_{\max}, \quad y_{\min} \le y \le y_{\max}, \quad 0.01\text{ m} \le z \le 0.40\text{ m}$$
- **Why height $z \in [0.01, 0.40]$ alone is not enough:** The plane $z \in [0.01, 0.40]$ extends infinitely across the entire room! Without the $(x, y)$ bounds of the tabletop, you would also capture chair backs, walls, and background furniture at that same elevation.

### 4.5 Minimal Isolated Snippet: Constructing a Frame & Transforming a Cloud
```python
import numpy as np
import open3d as o3d

def build_table_frame(table_normal: np.ndarray, table_center: np.ndarray) -> np.ndarray:
    # Ensure z_axis is a unit vector
    z_axis = table_normal / np.linalg.norm(table_normal)
    
    # Pick a non-collinear helper vector
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    
    # Construct orthonormal right-handed basis (x, y, z)
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Assemble 4x4 matrix T_scene_table
    T_scene_table = np.eye(4)
    T_scene_table[:3, 0] = x_axis
    T_scene_table[:3, 1] = y_axis
    T_scene_table[:3, 2] = z_axis
    T_scene_table[:3, 3] = table_center
    return T_scene_table

# Transform cloud into the table frame: p_table = inv(T_scene_table) @ p_scene
# pcd_in_table = copy.deepcopy(pcd_scene).transform(np.linalg.inv(T_scene_table))
```

---

## 5. Exercise 5: 3D Object Bounding Boxes (AABB vs. OBB)

### 5.1 Theoretical & Mathematical Comparison

Given a 3D point cloud of a single segmented object $\mathcal{O} = \{\mathbf{p}_i\}_{i=1}^M$:

#### 1. Axis-Aligned Bounding Box (AABB)
An AABB is defined strictly by the minimum and maximum coordinates along the current coordinate axes $(X, Y, Z)$:

$$\mathbf{p}_{\min} = \begin{bmatrix} \min_i x_i \\ \min_i y_i \\ \min_i z_i \end{bmatrix}, \quad \mathbf{p}_{\max} = \begin{bmatrix} \max_i x_i \\ \max_i y_i \\ \max_i z_i \end{bmatrix}$$

- **Extent (Dimensions):** $\mathbf{e}_{\text{AABB}} = \mathbf{p}_{\max} - \mathbf{p}_{\min} = [\Delta x, \Delta y, \Delta z]^T$.
- **Volume:** $V_{\text{AABB}} = \Delta x \cdot \Delta y \cdot \Delta z$.
- **Key Property:** Its edges are forced to remain parallel to the coordinate frame axes. Therefore, **if the object rotates in space, the AABB changes size and volume**! When a rectangular box (like a cereal box) is rotated $30^\circ$ around $Z$, the AABB must expand to enclose its diagonal corners, inflating with empty space.

#### 2. Oriented Bounding Box (OBB)
An OBB finds the intrinsic principal axes of the point cloud itself using **Principal Component Analysis (PCA)** (or minimum-volume convex hull orientation):
1. Computes the $3 \times 3$ covariance matrix $\mathbf{C}$ of the object's points (or of the triangles of its 3D convex hull).
2. Extracts the eigenvectors $\mathbf{R}_{\text{OBB}} = [\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3]$ as the local orientation of the box.
3. Projects the points onto $[\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3]$ to find the tightest local extents $\mathbf{e}_{\text{OBB}} = [l_1, l_2, l_3]^T$.

```
Top-Down View (XY Plane) of a Rotated Cereal Box:

        AABB (Axis-Aligned)                     OBB (Oriented with Object)
     ┌───────────────────────┐                        ╱╲
     │         ╱╲            │                       ╱  ╲
     │        ╱  ╲           │                      ╱    ╲
     │       ╱    ╲   Empty  │                     ╱      ╲
     │      ╱  Obj ╲  Space! │                     ╲  Obj  ╱
     │      ╲      ╱         │                      ╲    ╱
     │       ╲    ╱          │                       ╲  ╱
     └────────╲──╱───────────┘                        ╲╱
  (Volume inflates upon rotation!)           (Volume stays constant!)
```

### 5.2 Key Engineering Questions Explained
- **Which box changes when the object rotates by $30^\circ$ around $Z$?**
  - The **AABB** changes its extents and volume because its faces are locked to the coordinate system axes, whereas the object's diagonal projection onto $X$ and $Y$ changes with angle.
  - The **OBB** simply rotates along with the object; its intrinsic dimensions `extent` and `volume()` remain invariant!
- **Does the OBB always have a smaller volume than the AABB?**
  - Usually yes for elongated objects rotated relative to the axes, **but not always**! Open3D's `get_oriented_bounding_box()` aligns the box with the 3D PCA eigenvectors of the object's convex hull. If an object has uneven point density or asymmetric geometry, its 3D PCA axes can tilt in 3D, occasionally producing a slightly larger box than an AABB when the object was already axis-aligned.
- **Is the $Z$-axis of the OBB stays perpendicular to the table?**
  - **No!** Because 3D PCA computes eigenvectors in all three dimensions freely, the OBB's axes can pitch and roll relative to the table surface. By contrast, the AABB (computed in the table reference frame!) always has its bottom face strictly parallel to the tabletop and its $Z$-axis strictly perpendicular to the table.

### 5.3 API Mechanics & Minimal Isolated Snippet
- `pcd.get_axis_aligned_bounding_box() -> o3d.geometry.AxisAlignedBoundingBox`
  - `.get_extent()`: Returns `[dx, dy, dz]` array.
  - `.volume()`: Returns scalar volume.
  - `.color = [r, g, b]`: Sets wireframe line color.
- `pcd.get_oriented_bounding_box() -> o3d.geometry.OrientedBoundingBox`
  - `.extent`: Returns `[l1, l2, l3]` array.
  - `.R`: $3 \times 3$ rotation matrix of the box orientation.
  - `.volume()`: Returns scalar volume.
- Rotating a point cloud around its own center:
  - `R = pcd.get_rotation_matrix_from_xyz((rx, ry, rz))` (angles in **radians**).
  - `pcd.rotate(R, center=pcd.get_center())`.

```python
import copy
import numpy as np
import open3d as o3d

# Load a segmented object in the table frame
obj_pcd = o3d.io.read_point_cloud("object_0.ply")

# 1. Compute AABB and OBB
aabb = obj_pcd.get_axis_aligned_bounding_box()
aabb.color = (1.0, 0.0, 0.0)  # Red wireframe

obb = obj_pcd.get_oriented_bounding_box()
obb.color = (0.0, 1.0, 0.0)   # Green wireframe

print(f"AABB Extent: {aabb.get_extent()}, Volume: {aabb.volume():.6f} m³")
print(f"OBB  Extent: {obb.extent}, Volume: {obb.volume():.6f} m³")

# 2. Rotate object by 30 degrees around Z-axis about its center
obj_rot = copy.deepcopy(obj_pcd)
R_z30 = obj_rot.get_rotation_matrix_from_xyz((0.0, 0.0, np.radians(30.0)))
obj_rot.rotate(R_z30, center=obj_rot.get_center())

# 3. Recompute boxes after rotation
aabb_rot = obj_rot.get_axis_aligned_bounding_box()
aabb_rot.color = (1.0, 0.0, 0.0)

obb_rot = obj_rot.get_oriented_bounding_box()
obb_rot.color = (0.0, 1.0, 0.0)

o3d.visualization.draw_geometries([obj_rot, aabb_rot, obb_rot])
```
