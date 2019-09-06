import numpy as np
import copy
import open3d as o3d

def demo_crop_geometry(pcd):
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    # pcd = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    o3d.visualization.draw_geometries_with_editing([pcd])


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

def colored_icp(source,target,current_transformation):
    # print("1. Load two point clouds and show initial pose")
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.005]
    max_iter = [ 50]
    # print("3. Colored point cloud registration")
    for scale in range(1):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter, radius, scale])

        # print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = o3d.geometry.voxel_down_sample(source, radius)
        target_down = o3d.geometry.voxel_down_sample(target, radius)

        # print("3-2. Estimate normal.")
        o3d.geometry.estimate_normals(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius , max_nn=30))
        o3d.geometry.estimate_normals(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius , max_nn=30))

        # print("3-3. Applying colored point cloud registration")
        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=iter))
        current_transformation = result_icp.transformation
        # print(result_icp)
    # draw_registration_result_original_color(source, target,
    #                                         result_icp.transformation)

    return result_icp.transformation

if __name__ == "__main__":
    source = o3d.io.read_point_cloud("./whale1.ply").transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    target = o3d.io.read_point_cloud("./whale2.ply").transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    target2 = o3d.io.read_point_cloud("./whale3.ply").transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    # demo_crop_geometry(source)
    # demo_crop_geometry(target)
    trans = demo_manual_registration(source,target)

    # target_eff = o3d.geometry.PointCloud()
    # target_eff_Ar = []
    # target_eff_co = []
    # pick_points = np.asarray(target.points)
    # pick_color = np.asarray(target.colors)
    # for ind,point in enumerate(pick_points):
    #     # print(point)
    #     if point[2] < 0.230:
    #         target_eff_Ar.append(point)
    #         target_eff_co.append(pick_color[ind])
    # target_eff.points = o3d.utility.Vector3dVector(np.asarray(target_eff_Ar))
    # target_eff.colors = o3d.utility.Vector3dVector(np.asarray(target_eff_co))
    # o3d.visualization.draw_geometries([target_eff])
    # trans_refine = colored_icp(source,target,trans)

    # target += source.transform(trans)
    # target = o3d.geometry.voxel_down_sample(target, voxel_size=0.001)
    # o3d.io.write_point_cloud('registration.ply',target)

    target2 += source.transform(trans)
    target2= o3d.geometry.voxel_down_sample(target2, voxel_size=0.001)
    o3d.io.write_point_cloud('registration.ply',target2)

