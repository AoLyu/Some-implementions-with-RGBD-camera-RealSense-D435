import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time
from open3d import *
import os


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    align = rs.align(rs.stream.color)
    pipeline = rs.pipeline()
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming from file
    profile = pipeline.start(config)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # print( 'camera_intrinsic', intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
    pinhole_camera_intrinsic = PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

    geometrie_added = False
    vis = Visualizer()

    vis.create_window("Pointcloud")
    pointcloud = PointCloud()
    i = 0

    # Streaming loop
    try:
        while True:
            time_start = time.time()

            frames = pipeline.wait_for_frames()      
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            cv2.imshow('Color Stream', color_image1)

            depth = Image(depth_image)
            color = Image(color_image)

            rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
            pcd = create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

            if not pcd:
                continue

            pointcloud.clear()
            pointcloud += pcd

            if not geometrie_added:
                vis.add_geometry(pointcloud)
                geometrie_added = True

            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()


            depth_color_frame = rs.colorizer().colorize(depth_frame)

            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            cv2.imshow("Depth Stream", depth_color_image)
            key = cv2.waitKey(1)

            time_end = time.time()
            # print("FPS = {0}".format(int(1/(time_end - time_start))))

            if key & 0xFF == ord('s'):
                if not os.path.exists('./output/'): 
                    os.makedirs('./output')
                cv2.imwrite('./output/depth_'+str(i)+'.png',depth_image)
                cv2.imwrite('./output/color_'+str(i)+'.png',color_image1)
                write_point_cloud('./output/pointcloud_'+str(i)+'.pcd', pcd)
                print('No.'+str(i)+' shot is saved.' )
                i += 1

            # if pressed escape exit program
            elif key & 0xFF == ord('q') or key == 27 :
                cv2.destroyAllWindows()
                break

    finally:
        pass
