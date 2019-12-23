import _init_paths
import pyrealsense2 as rs 
from lib import rgbdTools,keyPoints,registration,template
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
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 6)
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
    temPoint3 = o3d.geometry.PointCloud()

    color_l = [(255,0,0),(255,0,255),(0,0,255),(0,255,255),(255,255,0),(96,96,96),(1,97,0),(227,207,87),(176,224,230),
            (106,90,205),(56,94,15),(61,89,171),(51,161,201),(178,34,34),(138,43,226)]

    pointcloud = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window("Pointcloud",640,480)
    geometrie_added = False

    i = 0

### Load template
    template_rgb = cv2.imread('./template/template.png')
    template_p = template.Template(template_rgb)
    _,_,temPoint3 = template_p.getPt()
    temFeatureList ,tem_xyr = template_p.getFeature()

    print("press 'a' three times to calculate the table plane coefficient.\n")
    
    
    # temFeatureList,tem_new_xyr = registration.extractFeatures(temPoint2,tem_old_xyr,n = 3)

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

        elif i == 1:
            a,b,c,d = keyPoints.calculatePlane(RealSense,depth_image,xl,yl,rl)
            TablePlane.getParam(a,b,c,d)
            print('calculating plane done!')
            for ind,x in enumerate(xl):
                x,y,z = rgbdTools.getPosition(RealSense,depth_image,yl[ind],xl[ind])
                if abs(TablePlane.a * x + TablePlane.b * y + TablePlane.c * z + TablePlane.d) / (TablePlane.a ** 2 + TablePlane.b**2 + TablePlane.c**2)**0.5 < 0.008:
                    cv2.circle(currentImage, (xl[ind],yl[ind]), rl[ind], (0, 255, 0), -1)
                    old_xyr.append([xl[ind],yl[ind],rl[ind]])

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
                i+=1
            elif i > 1:
                Pt2 = []
                Pt3 = []
                Pt4 = []
                Color4 = []
                # print('old_xyr:')
                # print(old_xyr)
                for xyr in old_xyr:
                    x2,y2,z2 = rgbdTools.getPosition(RealSense,depth_image,xyr[1],xyr[0])
                    Pt2.append([x2,y2,z2])
                    p_l = keyPoints.pointInRadius(xyr[0],xyr[1], 2)
                    for p in p_l:
                        m2,n2 = p
                        x,y,z = rgbdTools.getPosition(RealSense,depth_image,m2,n2)
                        Pt3.append([x,y,z])
                # print('Pt2')
                # print(Pt2)
                Point2.points = o3d.utility.Vector3dVector(np.array(Pt2))
                Point3.points = o3d.utility.Vector3dVector(np.array(Pt3))
                # o3d.visualization.draw_geometries([Point2])
                FeatureList,new_xyr = registration.extractFeatures(Point2,old_xyr,n = 3)

                print("press 'a' to get next view.\n")
                print("or press 's' to save current registrated pointcloud.\n")
                if 1:
                    # print(preFeatureList)
                    # print(1)
                    # print(FeatureList)
                    print("feature of view {} is extracted.".format(i-2))
                    Trans_init = None
                    Trans_init,ind_l = registration.fastMatch(FeatureList,temFeatureList)
                    # print(ind_l)
                    currentImage2 = color_image1.copy()
                    templateImage = template_rgb .copy()
                    for ci,pi in enumerate(ind_l):
                        sx = new_xyr[pi[0]][0]
                        sy = new_xyr[pi[0]][1]
                        sr = new_xyr[pi[0]][2]
                        cv2.circle(currentImage2,(sx,sy),sr,color_l[ci%len(color_l)],-1)
                        tx = tem_xyr[pi[1]][0]
                        ty = tem_xyr[pi[1]][1]
                        tr = tem_xyr[pi[1]][2]
                        cv2.circle(templateImage,(tx,ty),tr,color_l[ci%len(color_l)],-1)
                    templateImage = cv2.resize(templateImage,(480,480))
                    cv2.imshow('key Point',np.hstack((currentImage2,templateImage)))
                    
                    for mm in range(0,480):
                        for nn in range(0,640):
                            x,y,z = rgbdTools.getPosition(RealSense,depth_image,mm,nn)
                            if  y < (TablePlane.a * x + TablePlane.c * z + TablePlane.d)/(-TablePlane.b) -0.007 and z > 0.15 and z < 0.45:
                                Pt4.append([x,y,z])
                                Color4.append(rgbdTools.getColor(color_image1,mm,nn))
                    Point4.points = o3d.utility.Vector3dVector(np.array(Pt4))
                    Point4.colors = o3d.utility.Vector3dVector(np.array(Color4))
                    # refine_result = registration.icp(Point3,temPoint3, Trans_init)

                    pointcloud.clear()
                    pointcloud = globalPointcloud + copy.deepcopy(Point4).transform(Trans_init)
                    # pointcloud = globalPointcloud + copy.deepcopy(Point4).transform(Trans_init)
                    pointcloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                    pointcloud = pointcloud.voxel_down_sample(voxel_size=0.001)
                    cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.0)
                    pointcloud = pointcloud.select_down_sample(ind)
                    vis.add_geometry(pointcloud)
                i+=1
                print("press 'a' to get next view.\n")
                print("or press 's' to get current registrated pointcloud.\n")

        elif key & 0xFF == ord('s') and i > 2:
            globalPointcloud += Point4.transform(Trans_init)
            globalPointcloudcopy = copy.deepcopy(globalPointcloud)
            globalPointcloudcopy.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            globalPointcloudcopy= globalPointcloudcopy.voxel_down_sample( voxel_size=0.001)
            cl, ind = globalPointcloudcopy.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.0)
            globalPointcloudcopy = globalPointcloudcopy.select_down_sample(ind)
            o3d.io.write_point_cloud('global.ply',globalPointcloudcopy)
            o3d.io.write_point_cloud('global.pcd',globalPointcloudcopy)
            print('global is saved!\n\n')
            print("press 'a' to get next view.\n")
            print("or press 's' to get current registrated pointcloud.\n")
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


