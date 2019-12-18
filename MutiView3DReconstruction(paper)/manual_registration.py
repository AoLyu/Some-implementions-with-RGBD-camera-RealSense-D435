import numpy as np
import copy
import open3d as o3d

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def demo_manual_registration(source,target):
    print("Demo for manual ICP")
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))


    draw_registration_result(source, target, trans_init)
    # print("")
    return trans_init


if __name__ == "__main__":
    source = o3d.io.read_point_cloud("./box1.ply")
    # source2 = o3d.io.read_point_cloud("./elephant4.ply").transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    target = o3d.io.read_point_cloud("./box2.ply")
    # target2 = o3d.io.read_point_cloud("./elephant3.ply").transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    # target2 = o3d.io.read_point_cloud("./box3.ply")

    trans = demo_manual_registration(source,target)

    target += source.transform(trans)
    target = target.voxel_down_sample( voxel_size=0.001)
    cl, ind = target.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)
    inlier_cloud =  target.select_down_sample(ind)

    o3d.io.write_point_cloud('registration.ply',inlier_cloud)
    o3d.io.write_point_cloud('registration.pcd',inlier_cloud)

