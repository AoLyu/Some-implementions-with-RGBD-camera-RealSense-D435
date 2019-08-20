import pyrealsense2 as rs
import numpy as np
import cv2
import time
from open3d import *
import os

if __name__=="__main__":

    bag_name = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    
    output_path = bag_name + '.bag'

    align = rs.align(rs.stream.color)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    config.enable_record_to_file(output_path) 

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # print(type(pinhole_camera_intrinsic))
    
    cv2.namedWindow('Color Stream', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE)

    geometrie_added = False
    vis = Visualizer()
    #vis.create_window("Pointcloud",640,480)
    vis.create_window("Pointcloud")
    pointcloud = PointCloud()
    i = 0

    try:
        while True:
            dt0 = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            # depth_frame = rs.hole_filling_filter().process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            cv2.imshow('Color Stream', color_image1)
            cv2.imshow('Depth Stream', depth_color_image )

            depth = Image(depth_image)
            color = Image(color_image)

            rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
            pcd = create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

            if not pcd:
                continue
                
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            # pcd = voxel_down_sample(pcd, voxel_size = 0.003)
            pointcloud.clear()
            pointcloud += pcd

            if not geometrie_added:
                vis.add_geometry(pointcloud)
                geometrie_added = True

            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
            process_time = time.time() - dt0
            # print("FPS = {0}".format(int(1/process_time.total_seconds())))
            key = cv2.waitKey(1)

            if key & 0xFF == ord('s'):
                if not os.path.exists('./output/'): 
                    os.makedirs('./output')
                cv2.imwrite('./output/depth_'+str(i)+'.png',depth_image)
                cv2.imwrite('./output/color_'+str(i)+'.png',color_image1)
                write_point_cloud('./output/pointcloud_'+str(i)+'.pcd', pcd)
                print('No.'+str(i)+' shot is saved.' )
                i += 1

            # Press esc or 'q' to close the image window
            elif key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                vis.destroy_window()

                break
    finally:
        pipeline.stop()

