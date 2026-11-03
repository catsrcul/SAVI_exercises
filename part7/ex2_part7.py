#! /usr/bin/env python3

import open3d as o3d
import numpy as np

viewpoint = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" :
	[
		{
			"boundingbox_max" : [ 6.5291471481323242, 34.024543762207031, 11.225864410400391 ],
			"boundingbox_min" : [ -39.714397430419922, -16.512752532958984, -1.9472264051437378 ],
			"field_of_view" : 60.0,
			"front" : [ 0.6296625482551379, -0.47870724265195458, 0.6118532921848252 ],
			"lookat" : [ -9.6640795078702766, 0.32735962932038221, 3.5805180410650137 ],
			"up" : [ -0.50457854106294409, 0.34684419755037732, 0.7906324041705004 ],
			"zoom" : 0.50320000000000009
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def main():
    print("Load a ply point cloud, print it, and render it")
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud('miguelriemoliveira SAVI_26-27 main Parte07/point_clouds/factory.ply')
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=viewpoint['trajectory'][0]['front'],
                                      lookat=viewpoint['trajectory'][0]['lookat'],
                                      up=viewpoint['trajectory'][0]['up'])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd],
                                      zoom=0.3412,
                                      front=viewpoint['trajectory'][0]['front'],
                                      lookat=viewpoint['trajectory'][0]['lookat'],
                                      up=viewpoint['trajectory'][0]['up'])

    # cutting the point cloud

    points = np.asarray(downpcd.points)


    # ---------------- testing the ground values ----------------


    # Mechanical operation: compute distribution to see where the floor peak ends
    z_values = np.asarray(downpcd.points)[:, 2]
    counts, bin_edges = np.histogram(z_values, bins=50)
    # Print elevation bins to identify the floor spike
    for i in range(10):  # Inspect the lowest 10 bins
        print(f"Z range [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]: {counts[i]} points")

    # Minimal test: isolate the candidates to visually verify what they are
    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=0.3,  # Thickness tolerance (meters) around the fitted plane
        ransac_n=3,  # Number of points to sample per trial (keep 3 for planes)
        num_iterations=1000  # Number of random trials
    )
    [a, b, c, d] = plane_model
    print(f"Detected Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(f"Normal vector: [{a:.2f}, {b:.2f}, {c:.2f}]")

    # 1. Isolate the ground plane (the inliers)
    ground_pcd = pcd.select_by_index(inlier_indices)
    ground_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red

    # 2. Isolate the rest of the scene (the outliers)
    scene_no_ground = pcd.select_by_index(inlier_indices, invert=True)
    # 3. Visualize both to verify the cut
    o3d.visualization.draw_geometries([ground_pcd, scene_no_ground])

    # ---------------- end of testing the ground values ----------------

    # Persist processed cloud to disk
    o3d.io.write_point_cloud("factory_no_floor.ply", scene_no_ground)

    o3d.visualization.draw_geometries([scene_no_ground],
                                      zoom=0.3412,
                                      front=viewpoint['trajectory'][0]['front'],
                                      lookat=viewpoint['trajectory'][0]['lookat'],
                                      up=viewpoint['trajectory'][0]['up'])



if __name__ == '__main__':
    main()