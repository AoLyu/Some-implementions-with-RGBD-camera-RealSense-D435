import _init_paths
from tools import rgbdTools
from tools import keyPoints
from tools import registration
import cv2
import numpy as np 
from scipy import optimize
import open3d as o3d 
import sys
import os 
import time

time_start = time.time()

path_to_rgbd = './bear/'

RealSense = rgbdTools.Camera(fx = 616.8676, fy = 617.0631, cx = 319.5701, cy = 233.0649)
TablePlane = keyPoints.Plane()
cloud_number = 8
voxel_size =0.0001
Point2_list = []
Point3_list = []
Point4_list = []

for j in range(cloud_number):
    print('Processing No.',j,'view .')
    Point2 = o3d.PointCloud()
    Point3 = o3d.PointCloud()
    Point4 = o3d.PointCloud()
    Color2 = []
    Pt2 = []
    Pt3 = []
    Pt4 = []
    
    img = cv2.imread(path_to_rgbd + 'color/color_{}.png'.format(j))
    rgb = np.array(img)
    depth = cv2.imread(path_to_rgbd + 'depth/depth_{}.png'.format(j),-1)
    depth = np.asarray(depth)
    xl,yl,rl = keyPoints.getCircles(img)

    if j == 0:
        a,b,c,d = keyPoints.calculatePlane(RealSense,depth,xl,yl,rl)
        TablePlane.getParam(a,b,c,d)

    for ind,x in enumerate(xl):
        x,y,z = rgbdTools.getPosition(RealSense,depth,yl[ind],xl[ind])
        if abs(TablePlane.a * x + TablePlane.b * y + TablePlane.c * z + TablePlane.d) / (TablePlane.a ** 2 + TablePlane.b**2 + TablePlane.c**2)**0.5 < 0.008:
            x,y,z = rgbdTools.getPosition(RealSense,depth,yl[ind],xl[ind])
            Pt3.append([x,y,z])
            p_l = keyPoints.pointInRadius(xl[ind],yl[ind], rl[ind])
            for p in p_l:
                m2,n2 = p
                x,y,z = rgbdTools.getPosition(RealSense,depth,m2,n2)

                Pt2.append([x,y,z])
    for mm in range(0,480):
        for nn in range(0,640):
            x,y,z = rgbdTools.getPosition(RealSense,depth,mm,nn)
            if  y < (TablePlane.a * x + TablePlane.c * z + TablePlane.d)/(-TablePlane.b) - 0.01 and z > 0.15 and z < 0.35:

                Pt4.append([x,y,z])

                Color2.append(rgbdTools.getColor(rgb,mm,nn))

    Point2.points = o3d.Vector3dVector(np.array(Pt2))
    Point3.points = o3d.Vector3dVector(np.array(Pt3))
    Point4.points = o3d.Vector3dVector(np.array(Pt4))
    Point4.colors = o3d.Vector3dVector(np.array(Color2))

    Point2_list.append(Point2)
    Point3_list.append(Point3)
    Point4_list.append(Point4)



global_pcd = o3d.geometry.PointCloud()
for cloud_i in range(cloud_number-1):
    print('regitrating NO.',cloud_i,'view .')
    source = Point3_list[cloud_i]
    target = Point3_list[cloud_i+1]

    sFeatureList = registration.extractFeatures(source,3)
    tFeatureList = registration.extractFeatures(target,3)

    Trans_init = registration.fastMatch(sFeatureList,tFeatureList,n_pair=3)

    sourcePt = Point2_list[cloud_i]
    targetPt = Point2_list[cloud_i+1]   

    refine_result = registration.icp(sourcePt,targetPt,Trans_init)

    sourcePt2 = Point4_list[cloud_i]
    targetPt2 = Point4_list[cloud_i+1]
    
    global_pcd += sourcePt2

    global_pcd.transform(refine_result)
    global_pcd = o3d.geometry.voxel_down_sample(global_pcd, voxel_size)
    o3d.visualization.draw_geometries([global_pcd])
    if cloud_i == (cloud_number-2):
        # lastPt = o3d.io.read_point_cloud('./Pt4/Pt_{}.pcd'.format(cloud_number-1))
        lastPt = Point4_list[cloud_number-1]
        global_pcd += lastPt
        global_pcd = o3d.geometry.voxel_down_sample(global_pcd, voxel_size)
        o3d.visualization.draw_geometries([global_pcd])  
        
o3d.io.write_point_cloud('global.pcd',global_pcd)

time_end = time.time()
print('total cost:',time_end - time_start)
