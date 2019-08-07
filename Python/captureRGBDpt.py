import pyrealsense2 as rs
import numpy as np
import cv2
import time
from open3d import *
import os

if __name__=="__main__":
    align = rs.align(rs.stream.color)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # print(type(pinhole_camera_intrinsic))
    
    geometrie_added = False
    vis = Visualizer()
    vis.create_window("Pointcloud",640,480)
    pointcloud = PointCloud()
    i = 0

    try:
        while True:
            time_start = time.time()
            pointcloud.clear()

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

            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth image', depth_image )

            depth = Image(depth_image)
            color = Image(color_image)

            rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
            pcd = create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            # pcd = voxel_down_sample(pcd, voxel_size = 0.003)

            pointcloud += pcd

            if not geometrie_added:
                vis.add_geometry(pointcloud)
                geometrie_added = True


            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
            time_end = time.time()

            key = cv2.waitKey(1)

            print("FPS = {0}".format(int(1/(time_end-time_start))))

            # press 's' to save current RGBD images and pointcloud.
            if key & 0xFF == ord('s'):
                if not os.path.exists('./output/'): 
                    os.makedirs('./output')
                cv2.imwrite('./output/depth_'+str(i)+'.png',depth_image)
                cv2.imwrite('./output/color_'+str(i)+'.png',color_image1)
                write_point_cloud('./output/pointcloud_'+str(i)+'.pcd', pcd)
                print('No.'+str(i) + ' shot is saved.' )
                i += 1

            
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                vis.destroy_window()

                break
    finally:
        pipeline.stop()


