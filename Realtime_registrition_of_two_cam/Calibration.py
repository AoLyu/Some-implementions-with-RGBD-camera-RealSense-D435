import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os
import sys
from tools import rgbdTools,registration,keyPoints


if __name__ == '__main__':

    resolution_width = 1280 # pixels
    resolution_height = 720 # pixels
    frame_rate = 15  # fps

    align = rs.align(rs.stream.color)
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    # rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    # rs_config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, frame_rate)
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
    cam1 = rgbdTools.Camera(intr1.fx, intr1.fy, intr1.ppx, intr1.ppy)
    print('cam1 intrinsics:')
    print(intr1.width, intr1.height, intr1.fx, intr1.fy, intr1.ppx, intr1.ppy)

    pipeline2 = rs.pipeline()
    rs_config.enable_device(connect_device[1])
    pipeline_profile2 = pipeline2.start(rs_config)

    intr2 = pipeline_profile2.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cam2 = rgbdTools.Camera(intr2.fx, intr2.fy, intr2.ppx, intr2.ppy)
    print('cam2 intrinsics:')
    print(intr2.width, intr2.height, intr2.fx, intr2.fy, intr2.ppx, intr2.ppy)

    i = 0
    try:
        while True:
            time_start = time.time()
            frames1 = pipeline1.wait_for_frames()
            frames2 = pipeline2.wait_for_frames()

            aligned_frames1 = align.process(frames1)
            aligned_frames2 = align.process(frames2)

            color_frame1 = aligned_frames1.get_color_frame()
            depth_frame1 = aligned_frames1.get_depth_frame()

            depth_frame1 = rs.decimation_filter(1).process(depth_frame1)
            depth_frame1 = rs.disparity_transform(True).process(depth_frame1)
            depth_frame1 = rs.spatial_filter().process(depth_frame1)
            depth_frame1 = rs.temporal_filter().process(depth_frame1)
            depth_frame1 = rs.disparity_transform(False).process(depth_frame1)
            # depth_frame1 = rs.hole_filling_filter().process(depth_frame1)

            color_image1 = np.asanyarray(color_frame1.get_data())
            depth_image1 = np.asanyarray(depth_frame1.get_data())
            # depth_color_frame1 = rs.colorizer().colorize(depth_frame1)
            # depth_color_image1 = np.asanyarray(depth_color_frame1.get_data())

            # infrared_frame1 = aligned_frames1.get_infrared_frame()

            # infrared_image1 = np.asanyarray(infrared_frame1.get_data())

            # infrared_frame2 = aligned_frames2.get_infrared_frame()

            # infrared_image2 = np.asanyarray(infrared_frame2.get_data())

            color_frame2 = aligned_frames2.get_color_frame()
            depth_frame2 = aligned_frames2.get_depth_frame()
            depth_frame2 = rs.decimation_filter(1).process(depth_frame2)
            depth_frame2 = rs.disparity_transform(True).process(depth_frame2)
            depth_frame2 = rs.spatial_filter().process(depth_frame2)
            depth_frame2 = rs.temporal_filter().process(depth_frame2)
            depth_frame2 = rs.disparity_transform(False).process(depth_frame2)
            # depth_frame2 = rs.hole_filling_filter().process(depth_frame2)

            color_image2 = np.asanyarray(color_frame2.get_data())
            depth_image2 = np.asanyarray(depth_frame2.get_data())
            # depth_color_frame2 = rs.colorizer().colorize(depth_frame2)
            # depth_color_image2 = np.asanyarray(depth_color_frame2.get_data())

            color1 = color_image1.copy()
            color2 = color_image2.copy()

            chessboard_found1, corners1 = cv2.findChessboardCorners(color1, (9, 6))
            cv2.drawChessboardCorners(color1,(9,6),corners1,chessboard_found1)

            chessboard_found2, corners2 = cv2.findChessboardCorners(color2, (9, 6))
            cv2.drawChessboardCorners(color2,(9,6),corners2,chessboard_found2)

            cv2.namedWindow('cam1_cam2', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('cam1_cam2', np.hstack((color1,color2)))

            key = cv2.waitKey(1)
            time_end = time.time()
            # print('FPS:',1/(time_end-time_start))

            if key & 0xFF == ord('s'):
                if not os.path.exists('./output/'): 
                    os.makedirs('./output')
                cv2.imwrite('./output/cam1_depth_'+str(i)+'.png',depth_image1)
                cv2.imwrite('./output/cam1_color_'+str(i)+'.png',color_image1)
                # cv2.imwrite('./output/cam1_infrared_'+str(i)+'.png',infrared_image1)
                cv2.imwrite('./output/cam2_depth_'+str(i)+'.png',depth_image2)
                cv2.imwrite('./output/cam2_color_'+str(i)+'.png',color_image2)
                # cv2.imwrite('./output/cam2_infrared_'+str(i)+'.png',infrared_image2)
                # write_point_cloud('./output/pointcloud_'+str(i)+'.pcd', pcd)
                print('No.'+str(i) + ' shot is saved.' )
                i += 1

            elif key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pipeline1.stop()
        pipeline2.stop()
    
