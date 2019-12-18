import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os
import sys
import open3d as o3d  
from tools import rgbdTools,registration




if __name__ == '__main__':

    chessBoard_num = 4

    resolution_width = 1280 # pixels
    resolution_height = 720 # pixels
    frame_rate = 15  # fps

    align = rs.align(rs.stream.color)
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    connect_device = []
    for d in rs.context().devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))

    if len(connect_device) < 2:
        print('Registrition needs two camera connected.But got one.')
        exit()
    
    pipeline1 = rs.pipeline()
    rs_config.enable_device(connect_device[0])
    pipeline_profile1 = pipeline1.start(rs_config)

    intr1 = pipeline_profile1.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_camera1_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr1.width, intr1.height, intr1.fx, intr1.fy, intr1.ppx, intr1.ppy)
    cam1 = rgbdTools.Camera(intr1.fx, intr1.fy, intr1.ppx, intr1.ppy)
    # print('cam1 intrinsics:')
    # print(intr1.width, intr1.height, intr1.fx, intr1.fy, intr1.ppx, intr1.ppy)

    pipeline2 = rs.pipeline()
    rs_config.enable_device(connect_device[1])
    pipeline_profile2 = pipeline2.start(rs_config)

    intr2 = pipeline_profile2.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_camera2_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr2.width, intr2.height, intr2.fx, intr2.fy, intr2.ppx, intr2.ppy)
    cam2 = rgbdTools.Camera(intr2.fx, intr2.fy, intr2.ppx, intr2.ppy)
    # print('cam2 intrinsics:')
    # print(intr2.width, intr2.height, intr2.fx, intr2.fy, intr2.ppx, intr2.ppy)

    print('Calculating Transformation Matrix:')
    cam1_point = []
    cam2_point = []
    for view in range(chessBoard_num):
        cam1_rgb = cv2.imread('./output/cam1_color_'+str(view)+'.png')
        cam1_rgb_array = np.asanyarray(cam1_rgb)
        cam1_depth = cv2.imread('./output/cam1_depth_'+str(view)+'.png',-1)
        cam1_depth_array = np.asanyarray(cam1_depth)
        cam2_rgb = cv2.imread('./output/cam2_color_'+str(view)+'.png')
        cam2_rgb_array = np.asanyarray(cam2_rgb)
        cam2_depth = cv2.imread('./output/cam2_depth_'+str(view)+'.png',-1)
        cam2_depth_array = np.asanyarray(cam2_depth)
        chessboard_found1, corners1 = cv2.findChessboardCorners(cam1_rgb, (9, 6))
        corners1 = np.asanyarray(corners1).squeeze()
        chessboard_found2, corners2 = cv2.findChessboardCorners(cam2_rgb, (9, 6))
        corners2 = np.asanyarray(corners2).squeeze()

        for p_2d in range(54):
            # point2d_pair.append([corners1[p_2d],corners2[53-p_2d]])
            m1 = int(round(corners1[p_2d][1]))
            n1 = int(round(corners1[p_2d][0]))
            if cam1_depth_array[m1,n1] > 0:
                x1,y1,z1 = rgbdTools.getPosition(cam1,cam1_depth_array,m1,n1)
            else:
                continue
            m2 = int(round(corners2[53-p_2d][1]))
            n2 = int(round(corners2[53-p_2d][0]))
            if cam2_depth_array[m2,n2] > 0:
                x2,y2,z2 = rgbdTools.getPosition(cam2,cam2_depth_array,m2,n2)
            else:
                continue
            cam1_point.append([x1,y1,z1])
            cam2_point.append([x2,y2,z2])                     

    Transformation = registration.rigid_transform_3D(np.asarray(cam1_point),np.asarray(cam2_point))
    print(Transformation)

    geometrie_added = False
    vis = o3d.visualization.Visualizer()
    vis.create_window("Pointcloud")
    pointcloud = o3d.geometry.PointCloud()

    try:
        time_beigin = time.time()
        while True:
            time_start = time.time()
            pointcloud.clear()

            frames1 = pipeline1.wait_for_frames()
            frames2 = pipeline2.wait_for_frames()

            aligned_frames1 = align.process(frames1)
            aligned_frames2 = align.process(frames2)

            color_frame1 = aligned_frames1.get_color_frame()
            depth_frame1 = aligned_frames1.get_depth_frame()

            color_image1 = np.asanyarray(color_frame1.get_data())
            depth_image1 = np.asanyarray(depth_frame1.get_data())

            color_frame2 = aligned_frames2.get_color_frame()
            depth_frame2 = aligned_frames2.get_depth_frame()

            color_image2 = np.asanyarray(color_frame2.get_data())
            depth_image2 = np.asanyarray(depth_frame2.get_data())

            depth_image11 = np.where((depth_image1 > 1000) | (depth_image1 < 0), 0 , depth_image1)
            depth_image22 = np.where((depth_image2 > 1000) | (depth_image2 < 0), 0 , depth_image2)

            depth1 = o3d.geometry.Image(depth_image11)
            color1 = o3d.geometry.Image(cv2.cvtColor(color_image1, cv2.COLOR_BGR2RGB))


            depth2 = o3d.geometry.Image(depth_image22)
            color2 = o3d.geometry.Image(cv2.cvtColor(color_image2, cv2.COLOR_BGR2RGB))

            rgbd1 = o3d.geometry.RGBDImage.create_rgbd_image_from_color_and_depth(color1, depth1, convert_rgb_to_intensity = False)
            pcd1 = o3d.geometry.PointCloud.create_point_cloud_from_rgbd_image(rgbd1, pinhole_camera1_intrinsic)

            rgbd2 = o3d.geometry.RGBDImage.create_rgbd_image_from_color_and_depth(color2, depth2, convert_rgb_to_intensity = False)
            pcd2 = o3d.geometry.PointCloud.create_point_cloud_from_rgbd_image(rgbd2, pinhole_camera2_intrinsic)

            time_now =time.time()
            if time_now - time_beigin < 2:
                pointcloud += pcd2
            else:
                pointcloud += (pcd1.transform(Transformation) + pcd2)
            pointcloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            if not geometrie_added:
                vis.add_geometry(pointcloud)
                geometrie_added = True

            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()

            time_end = time.time()

            print("FPS = {0}".format(int(1/(time_end-time_start))))

    finally:
        pipeline1.stop()
        pipeline2.stop()