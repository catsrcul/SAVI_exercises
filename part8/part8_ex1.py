#! /usr/bin/env python3

import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

viewpoint = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" :
	[
		{
			"boundingbox_max" : [ 3.42578125, 2.7008080406188961, 2.1036736965179443 ],
			"boundingbox_min" : [ -0.30914504957199096, 0.89872457432746877, 0.47970107650756844 ],
			"field_of_view" : 60.0,
			"front" : [ 0.6063875761035562, -0.51000974180897984, -0.6100689885637457 ],
			"lookat" : [ 1.6602244185920121, 2.1334357588271282, 1.4482009082746938 ],
			"up" : [ -0.50781240758706181, -0.83876888292844076, 0.19645182547283002 ],
			"zoom" : 0.44590000000000002
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def main():
    def draw_registration_result(source, target, transformation, name='open3d'):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                          zoom=0.4459,
                                          front=viewpoint['trajectory'][0]['front'],
                                          lookat=viewpoint['trajectory'][0]['lookat'],
                                          up=viewpoint['trajectory'][0]['up'],
                                          window_name=str(name)
                                          )

    # definind the redings of the point clouds ------------______________________
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, trans_init)

    # measuring alignment for the 2 point clouds
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)
    #
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)


    # using symmetric ICP
    for pcd in (source, target):
        pcd.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        pcd.orient_normals_consistent_tangent_plane(30)

    reg_sym = o3d.pipelines.registration.registration_symmetric_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationSymmetric(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    print(reg_sym)

    draw_registration_result(source, target, reg_sym.transformation, name='symetric icp')

if __name__ == '__main__':
    main()