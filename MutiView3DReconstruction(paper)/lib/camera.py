import _init_paths
import pyrealsense2 as rs 
import cv2
import numpy as np 
import open3d as o3d 
import copy
import sys
import os 
import time

class Realsense():
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.align_to_color = rs.align(rs.stream.color)
        self.pipe_profile = self.pipeline.start(self.config)
        self.intr = self.pipe_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy)
        
    def getCurrentData(self,mode='rgb+depth+pointcloud+coloredDepth'):
        self.frames = self.pipeline.wait_for_frames()
        self.align_frames = self.align_to_color.process(self.frames)
        self.depth_frame = self.align_frames.get_depth_frame()
        self.color_frame = self.align_frames.get_color_frame()
        self.depth_color_frame = rs.colorizer().colorize(self.depth_frame)
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_color_image = np.asanyarray(self.depth_color_frame.get_data())
        if mode in ['rgb+depth+pointcloud','rgb+depth+pointcloud+coloredDepth']:
            depth = o3d.geometry.Image(self.depth_image)
            color_RGB = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
            rgb = o3d.geometry.Image(color_RGB)
            rgbd =  o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity = False)
            self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_camera_intrinsic)
            self.pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        if mode=='rgb+depth':
            return self.color_image,self.depth_image
        elif mode=='rgb+depth+pointcloud+coloredDepth':
            return self.color_image,self.depth_image,self.pcd,self.depth_color_image
        else:
            raise Exception("there is only two mode:'rgb+depth','rgb+depth+pointcloud+coloredDepth'.you can extend it by modifing lib/camera.py")

    def get_intrin_extrin(self):
        self.frames = self.pipeline.wait_for_frames()
        self.depth_frame = self.frames.get_depth_frame()
        self.color_frame = self.frames.get_color_frame()
        self.dprofile = self.depth_frame.get_profile()
        self.cprofile = self.color_frame.get_profile()
        self.cvsprofile = rs.video_stream_profile(self.cprofile)
        self.dvsprofile = rs.video_stream_profile(self.dprofile)
        self.color_intrin = self.cvsprofile.get_intrinsics()
        print("color_intrin",self.color_intrin)
        self.depth_intrin = self.dvsprofile.get_intrinsics()
        print("depth_intrin",self.depth_intrin)
        self.extrin = self.dprofile.get_extrinsics_to(self.cprofile)
        print("extrin: ", self.extrin)
        self.depth_sensor = self.pipe_profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print('depth_scale', self.depth_scale)
        
    def stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    camera1 = Realsense()

    cv2.namedWindow('Color Stream', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE)
    geometrie_added = False
    vis = o3d.visualization.Visualizer()
    #vis.create_window("Pointcloud",640,480)
    vis.create_window("Pointcloud")
    pointcloud = o3d.geometry.PointCloud()

    while True:
        dt0 = time.time()
        rgb , depth ,pcd, coloredDepth = camera1.getCurrentData(mode='rgb+depth+pointcloud+coloredDepth') # 2 mode 'rgb+depth','rgb+depth+pointcloud+coloredDepth'
        if not pcd:
            continue
        pointcloud.clear()
        pointcloud+=pcd
        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added =True
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        cv2.imshow('Color Stream',rgb)
        cv2.imshow('Depth Stream',coloredDepth)
        dt1 = time.time()
        print('FPS:',1/(dt1-dt0))
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            vis.destroy_window()
            break

    camera1.stop()