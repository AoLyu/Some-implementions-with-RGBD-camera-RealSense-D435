import _init_paths
import pyrealsense2 as rs 
from tools import rgbdTools,keyPoints,registration
import cv2
import numpy as np 
from scipy import optimize
import open3d as o3d 
import copy
import sys
import os 
import time


if __name__=="__main__":
    align = rs.align(rs.stream.color)
    #align = rs.align(rs.stream.depth)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    RealSense = rgbdTools.Camera(fx = intr.fx, fy = intr.fy, cx =  intr.ppx, cy = intr.ppy)
    TablePlane = keyPoints.Plane()
    # print(type(pinhole_camera_intrinsic))
    
    cv2.namedWindow('Color Stream', cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE)

    time_start = time.time()

    globalPointcloud = o3d.geometry.PointCloud()
    Point2 = o3d.geometry.PointCloud()
    Point3 = o3d.geometry.PointCloud()
    Point4 = o3d.geometry.PointCloud()

    prePoint3 =  o3d.geometry.PointCloud()
    prePoint4 =  o3d.geometry.PointCloud()

    color_l = [(255,0,0),(255,0,255),(0,0,255),(0,255,255),(255,255,0)]

    pointcloud = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window("Pointcloud",640,480)
    geometrie_added = False

    # Trans_init = None
    # ind_l = None
    # refine_result = None
    i = 0

### Load template
    template_rgb = cv2.imread('./template/templateRGB.png')
    template_depth = cv2.imread('./template/templateDepth.png',-1)
    template_depth = cv2.blur(template_depth, (3, 3))
    temRGB = np.array(template_rgb)
    temDepth = np.array(template_depth)
    xl,yl,rl = keyPoints.getCircles(template_rgb)
    tem_old_xyr = []
    temPt2 = []
    temPt3 = []
    temPoint2 = o3d.geometry.PointCloud()
    temPoint3 = o3d.geometry.PointCloud()

    plane_flag = 0

    for indt,x in enumerate(xl):
        tem_old_xyr.append([xl[indt],yl[indt],rl[indt]])
    # print(tem_old_xyr)
    for xyr in tem_old_xyr:
        x2,y2,z2 = rgbdTools.getPosition(RealSense,temDepth,xyr[1],xyr[0])
        temPt2.append([x2,y2,z2])
        temp_l = keyPoints.pointInRadius(xyr[0],xyr[1], 1)
        for p in temp_l:
            m2,n2 = p
            x,y,z = rgbdTools.getPosition(RealSense,temDepth,m2,n2)
            temPt3.append([x,y,z])
    temPoint2.points = o3d.utility.Vector3dVector(np.array(temPt2))
    temPoint3.points = o3d.utility.Vector3dVector(np.array(temPt3))
    
    temFeatureList,tem_new_xyr = registration.extractFeatures(temPoint2,tem_old_xyr,n = 3)


    while True:
        # Pt2 = []
        Point2 = o3d.geometry.PointCloud()

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

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        xl,yl,rl = keyPoints.getCircles(color_image1)
        old_xyr = []
        currentImage = color_image1.copy()

        if i == 0:
            for ind,x in enumerate(xl):
                cv2.circle(currentImage, (xl[ind],yl[ind]), rl[ind], (0, 255, 0), -1)

        elif i == 1 and plane_flag ==0:
            a,b,c,d = keyPoints.calculatePlane(RealSense,depth_image,xl,yl,rl)
            TablePlane.getParam(a,b,c,d)
            plane_flag +=1
            print('calculating plane done!')
            print("press 'a' twice to load the first view of the objecct's point cloud.")
            for ind,x in enumerate(xl):
                x,y,z = rgbdTools.getPosition(RealSense,depth_image,yl[ind],xl[ind])
                if abs(TablePlane.a * x + TablePlane.b * y + TablePlane.c * z + TablePlane.d) / (TablePlane.a ** 2 + TablePlane.b**2 + TablePlane.c**2)**0.5 < 0.008:
                    cv2.circle(currentImage, (xl[ind],yl[ind]), rl[ind], (0, 255, 0), -1)

        elif i > 1:
            for ind,x in enumerate(xl):
                x,y,z = rgbdTools.getPosition(RealSense,depth_image,yl[ind],xl[ind])
                if abs(TablePlane.a * x + TablePlane.b * y + TablePlane.c * z + TablePlane.d) / (TablePlane.a ** 2 + TablePlane.b**2 + TablePlane.c**2)**0.5 < 0.008:
                    cv2.circle(currentImage, (xl[ind],yl[ind]), rl[ind], (0, 255, 0), -1)
                    old_xyr.append([xl[ind],yl[ind],rl[ind]])


        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        cv2.imshow('Color Stream',currentImage)    

        key = cv2.waitKey(1)

        if key & 0xFF == ord('p'):
            for ind,x in enumerate(xl):
                cv2.circle(currentImage, (xl[ind],yl[ind]), rl[ind], (0, 255, 0), -1)
            a,b,c,d = keyPoints.calculatePlane(RealSense,depth_image,xl,yl,rl)
            TablePlane.getParam(a,b,c,d)

        if key & 0xFF == ord('a'):
            print(i)
            if i <= 1:
                if i == 0:
                    print("press 'a' to calculate the plane coefficient.")
                # if i == 2:
                #     print("press 'a' to load the first view of the objecct's point cloud.")
                i+=1
            elif i > 1:
                Pt2 = []
                Pt3 = []
                Pt4 = []
                Color4 = []
                for xyr in old_xyr:
                    x2,y2,z2 = rgbdTools.getPosition(RealSense,depth_image,xyr[1],xyr[0])
                    Pt2.append([x2,y2,z2])
                    p_l = keyPoints.pointInRadius(xyr[0],xyr[1], 1)
                    for p in p_l:
                        m2,n2 = p
                        x,y,z = rgbdTools.getPosition(RealSense,depth_image,m2,n2)
                        Pt3.append([x,y,z])
                Point2.points = o3d.Vector3dVector(np.array(Pt2))
                Point3.points = o3d.Vector3dVector(np.array(Pt3))
                # o3d.visualization.draw_geometries([Point2])
                FeatureList,new_xyr = registration.extractFeatures(Point2,old_xyr,n = 3)
                if 1:
                    # print(preFeatureList)
                    # print(1)
                    # print(FeatureList)
                    print("feature of view {} is extracted.".format(i-2))
                    Trans_init = None
                    Trans_init,ind_l = registration.fastMatch(FeatureList,temFeatureList,n_pair=5)
                    # print(ind_l)
                    currentImage2 = color_image1.copy()
                    templateImage = temRGB.copy()
                    for ci,pi in enumerate(ind_l):
                        sx = new_xyr[pi[0]][0]
                        sy = new_xyr[pi[0]][1]
                        sr = new_xyr[pi[0]][2]
                        cv2.circle(currentImage2,(sx,sy),sr,color_l[ci],-1)
                        tx = tem_new_xyr[pi[1]][0]
                        ty = tem_new_xyr[pi[1]][1]
                        tr = tem_new_xyr[pi[1]][2]
                        cv2.circle(templateImage,(tx,ty),tr,color_l[ci],-1)
                    cv2.imshow('key Point',np.hstack((currentImage2,templateImage)))
                    
                    for mm in range(0,480):
                        for nn in range(0,640):
                            x,y,z = rgbdTools.getPosition(RealSense,depth_image,mm,nn)
                            if  y < (TablePlane.a * x + TablePlane.c * z + TablePlane.d)/(-TablePlane.b) - 0.005 and z > 0.15 and z < 0.35:
                                Pt4.append([x,y,z])
                                Color4.append(rgbdTools.getColor(color_image1,mm,nn))
                    Point4.points = o3d.Vector3dVector(np.array(Pt4))
                    Point4.colors = o3d.Vector3dVector(np.array(Color4))
                    refine_result = registration.icp(Point3,temPoint3, Trans_init)

                    pointcloud.clear()
                    pointcloud = globalPointcloud + copy.deepcopy(Point4).transform(refine_result)
                    # pointcloud = globalPointcloud + copy.deepcopy(Point4).transform(Trans_init)
                    pointcloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                    pointcloud = o3d.geometry.voxel_down_sample(pointcloud, voxel_size=0.0002)
                    vis.add_geometry(pointcloud)
                    print("press 's' to save current registration or rotate objects and press 'a' to continue registration.")
                    # refine_result = None
                    # refine_result = registration.icp(Point3,temPoint3, Trans_init)
                    # pointcloud.clear()
                    # pointcloud += copy.deepcopy(globalPointcloud).transform(refine_result)
                    # pointcloud += Point4
                    # pointcloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                    # vis.add_geometry(pointcloud)
                    # draw_registration_result(pointcloud,globalPointcloud,Point4,refine_result)
                i+=1

        elif key & 0xFF == ord('s') and i > 2:
            globalPointcloud += Point4.transform(refine_result)
            globalPointcloudcopy = copy.deepcopy(globalPointcloud)
            globalPointcloudcopy.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            globalPointcloudcopy= o3d.geometry.voxel_down_sample(globalPointcloudcopy, voxel_size=0.0002)
            o3d.io.write_point_cloud('global.ply',globalPointcloudcopy)
            print('global is saved!')
            preImage = color_image1.copy()
            preFeatureList = FeatureList.copy()
            pre_xyr = new_xyr.copy()
            prePoint3 = copy.deepcopy(Point3)
            # prePoint4 = copy.deepcopy(Point4)

        elif key & 0xFF == ord('z'):
            cv2.destroyWindow('key Point')

        # Press esc or 'q' to close the image window
        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            vis.destroy_window()
            break     

    pipeline.stop()


