# SAVI - Part 7: 3D Point Cloud Processing with Open3D (Theory, Engineering Intuition & Mechanics)

Welcome to Part 7! In this module, we transition from 2D pixel grids ($H \times W$) to unstructured 3D spatial data: **Point Clouds**.

This guide provides the physical intuition, mathematical background, and isolated Open3D/NumPy API mechanics to understand and solve all exercises in Part 7.

---

## 0. The Big Picture: Why 3D Point Clouds?

In 2D computer vision, an image is a dense, uniform, regular grid of pixels: every pixel has fixed neighbors (up, down, left, right).

In 3D perception (LiDAR, RGB-D sensors, photogrammetry):
- Data is represented as a set of $N$ points in Euclidean space: $\mathcal{P} = \{\mathbf{p}_i = [x_i, y_i, z_i]^T \in \mathbb{R}^3\}_{i=1}^N$.
- **Unstructured & Sparse:** There is no grid or raster. Points can be irregularly spaced, density varies with distance, and empty space contains no data.
- **Permutation Invariant:** Shuffling the order of points in array memory does not change the physical shape.

### The Pipeline of Scene Understanding
To segment a building into meaningful structures (ground, walls, roof, balconies), we follow a structured processing pipeline:

```
[Raw Point Cloud: factory.ply]
             │
             ▼
[Crop / Filter Outliers (e.g. Ground Removal)]
             │
             ▼
[Spatial Decimation: Voxel Grid Downsampling]
             │
             ▼
[Euclidean Clustering (DBSCAN): Object Isolation]
             │
             ▼
[Local Surface Geometry: Normal Estimation (PCA)]
             │
             ▼
[Orientation Filtering: Angular Thresholding (Dot Product)]
      ├── Horizontal Surfaces (Roof / Balconies / Floor)
      └── Vertical Surfaces (Walls / Facades)
             │
             ▼
[Guided RANSAC Plane Segmentation]
```

---

## 1. Exercise 1: Loading, Visualizing, and Controlling the 3D Camera

### 1.1 Physical Intuition & Coordinate System
Open3D uses a right-handed Cartesian coordinate system:
- $\mathbf{X}$ (red axis): typically horizontal/lateral.
- $\mathbf{Y}$ (green axis): vertical or forward depending on sensor convention.
- $\mathbf{Z}$ (blue axis): elevation or depth.

When visualizing large point clouds, setting a deterministic view is critical for reproducibility, automation, and automated screen capture.

### 1.2 Mathematical Formulation: Pinhole Camera Trajectory
A 3D camera viewpoint in Open3D is determined by intrinsic and extrinsic transformation matrices:
- **Extrinsic Matrix ($\mathbf{T}_{\text{world}\to\text{cam}} \in \mathbf{SE}(3)$):**
  $$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$
  where $\mathbf{R} \in \mathbf{SO}(3)$ is the $3 \times 3$ rotation matrix and $\mathbf{t} \in \mathbb{R}^3$ is the translation vector.
- Alternatively, view control can be parameterized by:
  - **Lookat Vector ($\mathbf{p}_{\text{target}} \in \mathbb{R}^3$):** The 3D focal center around which the camera orbits.
  - **Eye Position ($\mathbf{p}_{\text{cam}} \in \mathbb{R}^3$):** The 3D location of the camera optical center.
  - **Up Vector ($\mathbf{u} \in \mathbb{R}^3$):** Defines which direction points toward the top of the screen (typically $[0, 0, 1]^T$ or $[0, -1, 0]^T$).
  - **Field of View (FOV) & Zoom.**

In Open3D, pressing `Ctrl + C` during a visualizer session dumps the camera parameters (JSON format) directly to the system clipboard. You can save these parameters into a `.json` file and reload them programmatically.

### 1.3 API Mechanics & Minimal Isolated Snippet
- `o3d.io.read_point_cloud(filename: str) -> o3d.geometry.PointCloud`
- `o3d.visualization.Visualizer()`: Lower-level viewer offering granular event and viewpoint control.
- `vis.get_view_control()`: Accesses the camera controller.

```python
import open3d as o3d

# 1. Load point cloud from disk
pcd = o3d.io.read_point_cloud("factory.ply")

# 2. Simple interactive display
o3d.visualization.draw_geometries([pcd], window_name="Raw Factory PCD")

# 3. Deterministic viewpoint initialization via Visualizer:
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Factory Viewpoint", width=1280, height=720)
vis.add_geometry(pcd)

ctr = vis.get_view_control()
# Load saved viewpoint JSON (or set lookat, front, up, zoom manually):
# params = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
# ctr.convert_from_pinhole_camera_parameters(params)

vis.run()
vis.destroy_window()
```

---

## 2. Exercise 2: Ground Floor Removal

### 2.1 The Engineering Problem
The ground plane usually contains the largest concentration of points in terrestrial scans. Leaving it in the cloud interferes with:
1. Object clustering (the ground connects all separate objects like a giant electrical short-circuit).
2. Vertical facade extraction (ground points bias normal calculations near building bases).

### 2.2 Mathematical Approaches: Height Cutoff vs. Plane Fit
There are two primary methods to isolate and remove the floor:

#### Method A: Coordinate Altitude Thresholding (Height Slicing)
If the scanner's $Z$-axis is perpendicular to a flat floor:
$$\mathcal{P}_{\text{no\_floor}} = \{ \mathbf{p} = [x, y, z]^T \in \mathcal{P} \mid z > z_{\text{floor\_threshold}} \}$$
This is computationally trivial ($\mathcal{O}(N)$), but requires the floor to be strictly horizontal and aligned with the coordinate axes.

#### Method B: RANSAC Plane Removal
If the floor is tilted or uneven:
Fit a plane equation $ax + by + cz + d = 0$ using RANSAC. The floor points correspond to the largest consensus set whose normal is approximately vertical ($|c| \approx 1$). Points classified as floor inliers are then inverted:
$$\mathcal{P}_{\text{no\_floor}} = \mathcal{P} \setminus \mathcal{P}_{\text{floor\_inliers}}$$

### 2.3 API Mechanics & Minimal Isolated Snippet
- NumPy conversion: `np.asarray(pcd.points)` returns an $(N, 3)$ float64 array sharing memory with the Open3D geometry.
- `pcd.select_by_index(indices, invert=True)`: Extracts or crops points using integer indices.

```python
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("factory.ply")
points = np.asarray(pcd.points)

# Find minimum Z or inspect histogram to set threshold
z_min = np.min(points[:, 2])
floor_threshold = z_min + 0.5  # Adjust according to scene units (e.g. meters)

# Boolean masking to identify non-floor indices
non_floor_indices = np.where(points[:, 2] > floor_threshold)[0]
pcd_no_floor = pcd.select_by_index(non_floor_indices)

# Persist processed cloud to disk
o3d.io.write_point_cloud("factory_no_floor.ply", pcd_no_floor)
```

---

## 3. Exercise 3: Sequential RANSAC Plane Detection (and Its Limitations)

### 3.1 Mathematical Formulation of a 3D Plane
A 3D plane $\Pi$ is parameterized by 4 coefficients $\mathbf{\pi} = [a, b, c, d]^T$:

$$a x + b y + c z + d = 0 \quad \text{with} \quad \|\mathbf{n}\|_2 = \sqrt{a^2 + b^2 + c^2} = 1$$

where $\mathbf{n} = [a, b, c]^T$ represents the surface unit normal vector and $|d|$ is the orthogonal distance from the origin to the plane.

The perpendicular Euclidean distance from an arbitrary point $\mathbf{p}_i = [x_i, y_i, z_i]^T$ to plane $\Pi$ is:

$$D(\mathbf{p}_i, \Pi) = \frac{|a x_i + b y_i + c z_i + d|}{\sqrt{a^2 + b^2 + c^2}} = |a x_i + b y_i + c z_i + d|$$

### 3.2 RANSAC (Random Sample Consensus) Mechanism
To identify the dominant plane in an unorganized point cloud:
1. **Hypothesis Generation:** Randomly draw 3 non-collinear points $(\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3)$. Formulate plane normal:
   $$\mathbf{n} = (\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1), \quad \mathbf{n} \leftarrow \frac{\mathbf{n}}{\|\mathbf{n}\|_2}$$
   $$d = -\mathbf{n} \cdot \mathbf{p}_1$$
2. **Consensus Evaluation:** Measure $D(\mathbf{p}_i, \Pi)$ for all $N$ points. Count inliers satisfying:
   $$D(\mathbf{p}_i, \Pi) \le \delta_{\text{distance\_threshold}}$$
3. **Iteration & Selection:** Repeat for $K$ iterations. Retain plane candidate with the highest inlier count.

### 3.3 Sequential Multi-Plane Segmentation
To detect multiple planes:
1. Run RANSAC on cloud $\mathcal{P}_k$.
2. Segment inlier points $\mathcal{I}_k$ (paint them with a unique color).
3. Update remainder cloud: $\mathcal{P}_{k+1} \leftarrow \mathcal{P}_k \setminus \mathcal{I}_k$.
4. Repeat until remaining points fall below a minimum cluster size.

```
       [Point Cloud P]
              │
              ▼
    ┌──────────────────┐
    │  RANSAC Plane 1  │
    └─────────┬────────┘
              ├──► Inliers 1 (Colored, e.g. Green)
              ▼
    [Remaining Points P \ I1]
              │
              ▼
    ┌──────────────────┐
    │  RANSAC Plane 2  │
    └─────────┬────────┘
              ├──► Inliers 2 (Colored, e.g. Pink)
              ▼
    [Remaining Points P \ I2] ──► ...
```

### 3.4 Why Naive Sequential RANSAC Fails (Engineering Analysis)
As observed in Exercise 3 (the incorrect yellow and orange planes):
- **Spurious Geometric Alignments:** In complex scenes, 3 randomly sampled points might pick one point on a wall, one on a roof, and one on a tree. If a tilted plane passing through them happens to slice through unrelated structures, it accumulates a high inlier count by pure chance.
- **Ignorance of Local Surface Geometry:** Standard RANSAC only measures **point-to-plane distance** $D(\mathbf{p}_i, \Pi)$. It does **not** check whether the local surface orientation at that point matches the plane's normal!
- **Solution:** Guide RANSAC using **surface normals** and **clustering**, ensuring only points belonging to the same spatial object and surface orientation are considered.

### 3.5 API Mechanics & Minimal Isolated Snippet
- `pcd.segment_plane(distance_threshold, ransac_n, num_iterations)`:
  - `distance_threshold`: Maximum distance $\delta$ for inlier classification.
  - `ransac_n=3`: Minimal number of points to define a plane.
  - `num_iterations`: Number of random trials.
  - Returns: `(plane_model, inlier_indices)` where `plane_model = [a, b, c, d]`.

```python
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("factory_no_floor.ply")
remaining_pcd = pcd
planes = []

colors = [[0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0.5, 0]]  # Green, Pink, Yellow, Orange

for i in range(4):
    plane_model, inliers = remaining_pcd.segment_plane(
        distance_threshold=0.2,
        ransac_n=3,
        num_iterations=1000
    )
    if len(inliers) < 100:
        break
        
    inlier_cloud = remaining_pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color(colors[i % len(colors)])
    planes.append(inlier_cloud)
    
    # Strip detected plane from cloud
    remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

o3d.visualization.draw_geometries(planes + [remaining_pcd])
```

---

## 4. Exercise 4: Voxel Grid Downsampling & Euclidean Clustering (DBSCAN)

### 4.1 Voxel Grid Downsampling
Raw LiDAR scans can contain millions of points. Points close to the scanner are densely packed, while distant points are sparse.

#### Mathematical Operation:
1. The 3D bounding box is partitioned into regular cubic voxels of side length $v_s$ (e.g., $v_s = 0.3 \text{ m}$):
   $$\mathbf{k} = \begin{bmatrix} k_x \\ k_y \\ k_z \end{bmatrix} = \left\lfloor \frac{\mathbf{p}_i - \mathbf{p}_{\min}}{v_s} \right\rfloor$$
2. All $M_k$ points falling into voxel $\mathbf{k}$ are averaged into a single centroid:
   $$\mathbf{p}_{\text{voxel}} = \frac{1}{M_k} \sum_{j=1}^{M_k} \mathbf{p}_j$$

**Benefits:**
- Drastically reduces point count (e.g., $90\%$ reduction) without altering global geometric topology.
- Regularizes point density across distance.
- Significantly accelerates subsequent normal estimation and clustering.

### 4.2 DBSCAN Clustering (Density-Based Spatial Clustering of Applications with Noise)
To isolate the main factory building from peripheral trees, fences, and scanning noise, we group points into connected components based on spatial proximity.

#### Algorithm Mechanics:
- **$\epsilon$ (`eps`):** Radius of the neighborhood sphere around point $\mathbf{p}_i$.
- **$\text{MinPts}$ (`min_points`):** Minimum number of points required inside the $\epsilon$-sphere for $\mathbf{p}_i$ to be considered a **core point**.
- Points within $\epsilon$ of core points are recursively chained into clusters. Points that cannot be chained are flagged as noise (`cluster_id = -1`).

```
    ● (Core point: ≥ MinPts in ε-ball)
   / \
  ●───● (Density-connected cluster)
  
          x (Noise point: isolated, < MinPts)
```

By counting the number of points in each cluster, the building corresponds to the largest cluster.

### 4.3 API Mechanics & Minimal Isolated Snippet
- `pcd.voxel_down_sample(voxel_size: float)`
- `pcd.cluster_dbscan(eps: float, min_points: int, print_progress=False)`: Returns a list of integer cluster labels (`-1` indicates noise).

```python
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("factory_no_floor.ply")

# 1. Downsample to 0.3m grid
down_pcd = pcd.voxel_down_sample(voxel_size=0.3)

# 2. Euclidean Clustering via DBSCAN
labels = np.array(down_pcd.cluster_dbscan(eps=0.8, min_points=10, print_progress=True))

# 3. Find the largest cluster (the main building)
# Exclude noise label (-1)
valid_labels = labels[labels >= 0]
largest_cluster_id = np.bincount(valid_labels).argmax()

building_indices = np.where(labels == largest_cluster_id)[0]
building_pcd = down_pcd.select_by_index(building_indices)

o3d.visualization.draw_geometries([building_pcd])
```

---

## 5. Exercise 5: Surface Normal Estimation via Local PCA

### 5.1 What is a Surface Normal?
A surface normal $\mathbf{n} = [n_x, n_y, n_z]^T$ is a unit vector perpendicular to the local tangent plane of the underlying surface at point $\mathbf{p}_i$:

$$\|\mathbf{n}\|_2 = 1, \quad \mathbf{n} \cdot \mathbf{v}_{\text{tangent}} = 0$$

Point clouds do not store surface connectivity (triangles/polygons). Therefore, normals must be estimated statistically from local neighborhoods.

### 5.2 Mathematical Formulation: Covariance Matrix & Eigen-Decomposition
For each point $\mathbf{p}_i$, we gather its $K$ nearest spatial neighbors $\mathcal{N}_i = \{\mathbf{q}_1, \dots, \mathbf{q}_K\}$:
1. Compute the local centroid:
   $$\bar{\mathbf{p}} = \frac{1}{K} \sum_{j=1}^K \mathbf{q}_j$$
2. Construct the $3 \times 3$ sample covariance matrix $\mathbf{C}$:
   $$\mathbf{C} = \frac{1}{K} \sum_{j=1}^K (\mathbf{q}_j - \bar{\mathbf{p}})(\mathbf{q}_j - \bar{\mathbf{p}})^T$$
3. Perform Singular Value / Eigen-decomposition:
   $$\mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k, \quad \text{with } \lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$$
   - Eigenvectors $\mathbf{v}_1, \mathbf{v}_2$ correspond to the directions of largest local dispersion (the tangent plane).
   - The eigenvector $\mathbf{v}_3$ corresponding to the **smallest eigenvalue ($\lambda_3$)** represents the direction of least variance—which is precisely the **surface normal vector**:
     $$\mathbf{n}_i = \mathbf{v}_3$$

### 5.3 Normal Orientation Consistency
Eigen-decomposition determines the normal line, but has a sign ambiguity ($\pm \mathbf{v}_3$). To ensure normals point consistently toward the scanner:
$$\mathbf{n}_i \leftarrow -\mathbf{n}_i \quad \text{if} \quad \mathbf{n}_i \cdot (\mathbf{p}_{\text{sensor}} - \mathbf{p}_i) < 0$$

### 5.4 API Mechanics & Minimal Isolated Snippet
- `pcd.estimate_normals(search_param=...)`
  - `o3d.geometry.KDTreeSearchParamHybrid(radius: float, max_nn: int)`: Hybrid search bounded by radius and maximum neighbor count.
- `pcd.orient_normals_consistent_tangent_plane(k: int)`

```python
import open3d as o3d

# Estimate surface normals for downsampled building
building_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
)

# Optional: orient normals consistently
building_pcd.orient_normals_to_align_with_direction(orientation_reference=[0.0, 0.0, 1.0])

# Visualize point cloud with normal vectors rendered as lines
o3d.visualization.draw_geometries([building_pcd], point_show_normal=True)
```

---

## 6. Exercises 6 & 7: Orientation Filtering via Vector Dot Products

### 6.1 Mathematical Formulation of Orientation Angle ($\theta$)
Let $\mathbf{n} = [n_x, n_y, n_z]^T$ be the normalized unit normal vector at a given point ($\|\mathbf{n}\|_2 = 1$).
Let $\mathbf{z} = [0, 0, 1]^T$ be the vertical unit vector along the $Z$-axis ($\|\mathbf{z}\|_2 = 1$).

From the geometric definition of the inner (dot) product:
$$\mathbf{n} \cdot \mathbf{z} = \|\mathbf{n}\|_2 \|\mathbf{z}\|_2 \cos(\theta) = 1 \cdot 1 \cdot \cos(\theta) = n_z$$

Solving for the angle $\theta \in [0, \pi]$:

$$\theta = \arccos\left(\frac{\mathbf{n} \cdot \mathbf{z}}{\|\mathbf{n}\|_2 \|\mathbf{z}\|_2}\right) = \arccos(n_z)$$

In practice, because normal vectors might point upward or downward ($+Z$ or $-Z$), we evaluate the absolute alignment with the vertical axis:
$$|\cos(\theta)| = |n_z|$$

```
Vertical Vector z = [0, 0, 1]
         ▲
         │  θ ≈ 0° (or 180°)  ──► Horizontal Surface (Roofs, Balconies, Floors)
         │  |nz| ≈ 1
         │
         │  θ ≈ 90°           ──► Vertical Surface (Walls, Facades)
         │  |nz| ≈ 0
         └──────────────────► Horizontal Plane
```

---

### 6.2 Exercise 6: Horizontal Surfaces (Roofs & Balconies)
A horizontal surface has a normal vector pointing straight up or down.
- $\theta \approx 0^\circ$ or $\theta \approx 180^\circ$.
- $|n_z| \approx 1$.

Given an angular tolerance $\Delta \theta$ (e.g., $15^\circ \approx 0.26 \text{ rad}$):
$$\theta \le \Delta \theta \quad \text{or} \quad \theta \ge (180^\circ - \Delta \theta)$$
Equivalently:
$$|n_z| \ge \cos(\Delta \theta)$$

For $\Delta \theta = 15^\circ$, $\cos(15^\circ) \approx 0.966$. Any point with $|n_z| \ge 0.966$ belongs to a horizontal plane.

```python
import numpy as np

normals = np.asarray(building_pcd.normals) # Shape: (N, 3)
nz = normals[:, 2]

# Threshold for horizontal surfaces: within 15 degrees of vertical z-axis
angle_tol_rad = np.radians(15.0)
horizontal_mask = np.abs(nz) >= np.cos(angle_tol_rad)

horizontal_indices = np.where(horizontal_mask)[0]
horizontal_pcd = building_pcd.select_by_index(horizontal_indices)
horizontal_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Green
```

---

### 6.3 Exercise 7: Vertical Surfaces (Walls & Facades)
A vertical wall has a normal vector that lies in the $XY$-plane (orthogonal to the $Z$-axis).
- $\theta \approx 90^\circ$.
- $n_z \approx 0$.

Given angular tolerance $\Delta \theta$ around $90^\circ$ (e.g., $75^\circ \le \theta \le 105^\circ$):
$$|n_z| \le \sin(\Delta \theta)$$

For $\Delta \theta = 15^\circ$, $\sin(15^\circ) \approx 0.259$. Any point with $|n_z| \le 0.259$ belongs to a vertical wall.

```python
# Threshold for vertical surfaces: within 15 degrees of horizontal plane
vertical_mask = np.abs(nz) <= np.sin(angle_tol_rad)

vertical_indices = np.where(vertical_mask)[0]
vertical_pcd = building_pcd.select_by_index(vertical_indices)
vertical_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # Blue
```

---

## 7. Exercise 8: Guided RANSAC Plane Segmentation

### 7.1 Why Normal-Guided RANSAC Solves the Problem
Recall the failure mode in Exercise 3: RANSAC randomly sampled across the entire building, finding slanted diagonal planes that grouped wall points, balcony points, and trees together.

In **Guided RANSAC**:
1. We **partition the point cloud first** using surface orientation:
   - Run RANSAC on `horizontal_pcd` $\to$ Detects roofs, terraces, and balcony floors with zero risk of accidentally fitting a wall.
   - Run RANSAC on `vertical_pcd` $\to$ Detects individual exterior facades and walls with zero risk of grouping roofs or floors.
2. In addition, during RANSAC fitting, we can verify that every candidate plane's normal $\mathbf{n}_{\Pi}$ matches the pre-filtered orientation of its points.

### 7.2 Multi-Plane Guided Extraction Strategy
For vertical walls:
1. Start with `vertical_pcd`.
2. Run RANSAC plane segmentation:
   - Model: $ax + by + cz + d = 0$.
   - Verify that $|c| \le \sin(15^\circ)$ (ensuring the fitted plane is vertical).
3. Store the inlier wall plane, assign a distinct color, and remove its inliers from `vertical_pcd`.
4. Repeat until all primary walls are segmented.

For horizontal surfaces:
1. Start with `horizontal_pcd`.
2. Run RANSAC plane segmentation:
   - Verify that $|c| \ge \cos(15^\circ)$ (ensuring the fitted plane is horizontal).
3. Segment the roof and terraces.

### 7.3 Minimal Isolated Snippet
```python
import open3d as o3d
import numpy as np

def extract_guided_planes(pcd, max_planes=5, dist_thresh=0.15, min_points=200):
    remaining = pcd
    detected_planes = []
    
    # Generate distinct random colors
    np.random.seed(42)
    colors = np.random.uniform(0.2, 0.9, size=(max_planes, 3))
    
    for i in range(max_planes):
        if len(remaining.points) < min_points:
            break
            
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=2000
        )
        
        if len(inliers) < min_points:
            break
            
        plane_cloud = remaining.select_by_index(inliers)
        plane_cloud.paint_uniform_color(colors[i])
        detected_planes.append(plane_cloud)
        
        # Remove inliers to discover subsequent planes
        remaining = remaining.select_by_index(inliers, invert=True)
        
    return detected_planes

# 1. Segment vertical walls from the vertical filtered subset
wall_planes = extract_guided_planes(vertical_pcd, max_planes=6, dist_thresh=0.2)

# 2. Segment horizontal roofs/terraces from the horizontal filtered subset
roof_planes = extract_guided_planes(horizontal_pcd, max_planes=4, dist_thresh=0.15)

# 3. Visualize complete segmented model
o3d.visualization.draw_geometries(wall_planes + roof_planes)
```

---

## 8. Practical Implementation Checklist

When implementing the code files for Part 7:

1. **Exercise 1 (`p7_ex1.py`):**
   - Load `factory.ply` using `o3d.io.read_point_cloud`.
   - Setup a `Visualizer` window and save/load camera parameters so it always starts from the specified vantage point.
2. **Exercise 2 (`p7_ex2.py`):**
   - Inspect $Z$ distribution via NumPy and crop out the ground points. Save the result as `factory_no_floor.ply`.
3. **Exercise 3 (`p7_ex3.py`):**
   - Run sequential `segment_plane` iterations to reproduce the raw multi-plane detection and observe the spurious diagonal planes.
4. **Exercise 4 (`p7_ex4.py`):**
   - Downsample using `voxel_down_sample(voxel_size=0.3)`.
   - Cluster with `cluster_dbscan` to isolate the building geometry from peripheral scanning artifacts.
5. **Exercise 5 (`p7_ex5.py`):**
   - Estimate normals using `estimate_normals(KDTreeSearchParamHybrid(radius=1.0, max_nn=30))`.
6. **Exercises 6 & 7 (`p7_ex6.py`, `p7_ex7.py`):**
   - Extract the $Z$ component of the unit normal vector array ($n_z$).
   - Apply $|n_z| \ge \cos(15^\circ)$ for horizontal surfaces (green).
   - Apply $|n_z| \le \sin(15^\circ)$ for vertical surfaces (blue).
7. **Exercise 8 (`p7_ex8.py`):**
   - Combine the orientation-filtered subsets with RANSAC to reliably extract all walls, roof, and balconies without cross-talk or diagonal artifacts.
