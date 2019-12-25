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

if __name__ == "__main__":

    RealSense = rgbdTools.Camera(fx = 616.8676, fy = 617.0631, cx =  319.5701, cy = 233.0649)
    TablePlane = keyPoints.Plane()

    if not os.path.exists("output"):
        os.makedirs("output")

    globalPointcloud = o3d.geometry.PointCloud()
    globalPointcloud1 = o3d.geometry.PointCloud()
    globalPointcloud2 = o3d.geometry.PointCloud()

    Point2 = o3d.geometry.PointCloud()
    Point3 = o3d.geometry.PointCloud()
    Point4 = o3d.geometry.PointCloud()

    prePoint3 =  o3d.geometry.PointCloud()
    prePoint4 =  o3d.geometry.PointCloud()
    temPoint3 = o3d.geometry.PointCloud()

    pointcloud = o3d.geometry.PointCloud()

    template_rgb = cv2.imread('./template/template.png')
    template_p = template.Template(template_rgb)
    _,_,temPoint3 = template_p.getPt()
    temFeatureList ,tem_xyr = template_p.getFeature()

    for top_i in range(6):
        color_image = cv2.imread("data1/rgb_{}.png".format(top_i))
        depth_image = cv2.imread("data1/depth_{}.png".format(top_i),-1)

        xl,yl,rl = keyPoints.getCircles(color_image)
        old_xyr = []
        currentImage = color_image.copy()

        if top_i == 0:
            a,b,c,d = keyPoints.calculatePlane(RealSense,depth_image,xl,yl,rl)
            TablePlane.getParam(a,b,c,d)

        for ind,x in enumerate(xl):
            x,y,z = rgbdTools.getPosition(RealSense,depth_image,yl[ind],xl[ind])
            if abs(TablePlane.a * x + TablePlane.b * y + TablePlane.c * z + TablePlane.d) / (TablePlane.a ** 2 + TablePlane.b**2 + TablePlane.c**2)**0.5 < 0.008:
                old_xyr.append([xl[ind],yl[ind],rl[ind]])

        Pt2 = []
        Pt3 = []
        Pt4 = []
        Color4 = []

        for xyr in old_xyr:
            x2,y2,z2 = rgbdTools.getPosition(RealSense,depth_image,xyr[1],xyr[0])
            Pt2.append([x2,y2,z2])
            p_l = keyPoints.pointInRadius(xyr[0],xyr[1], 2)
            for p in p_l:
                m2,n2 = p
                x,y,z = rgbdTools.getPosition(RealSense,depth_image,m2,n2)
                Pt3.append([x,y,z])

        Point2.points = o3d.utility.Vector3dVector(np.array(Pt2))
        Point3.points = o3d.utility.Vector3dVector(np.array(Pt3))
        FeatureList,new_xyr = registration.extractFeatures(Point2,old_xyr,n = 3)

        Trans_init,ind_l = registration.fastMatch(FeatureList,temFeatureList)

        for mm in range(0,480):
            for nn in range(0,640):
                x,y,z = rgbdTools.getPosition(RealSense,depth_image,mm,nn)
                if  y < (TablePlane.a * x + TablePlane.c * z + TablePlane.d)/(-TablePlane.b) -0.007 and z > 0.15 and z < 0.45:
                    Pt4.append([x,y,z])
                    Color4.append(rgbdTools.getColor(color_image,mm,nn))

        Point4.points = o3d.utility.Vector3dVector(np.array(Pt4))
        Point4.colors = o3d.utility.Vector3dVector(np.array(Color4))

        Point4 = Point4.voxel_down_sample(voxel_size=0.001)
        cl, ind = Point4.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.0)
        Point4 = Point4.select_down_sample(ind)

        globalPointcloud1 += Point4.transform(Trans_init)

        print("top view_{} is added.".format(top_i))

    # globalPointcloud1 = globalPointcloud1.voxel_down_sample(voxel_size=0.001)
    # cl, ind = globalPointcloud1.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.0)
    # globalPointcloud1 = globalPointcloud1.select_down_sample(ind)

    o3d.io.write_point_cloud("./output/top.pcd",globalPointcloud1)
    o3d.io.write_point_cloud("./output/top.ply",globalPointcloud1)
    print("top part is generated.")


    for bottom_i in range(5):
        color_image = cv2.imread("data2/rgb_{}.png".format(bottom_i))
        depth_image = cv2.imread("data2/depth_{}.png".format(bottom_i),-1)

        xl,yl,rl = keyPoints.getCircles(color_image)
        old_xyr = []
        currentImage = color_image.copy()

        if bottom_i == 0:
            a,b,c,d = keyPoints.calculatePlane(RealSense,depth_image,xl,yl,rl)
            TablePlane.getParam(a,b,c,d)

        for ind,x in enumerate(xl):
            x,y,z = rgbdTools.getPosition(RealSense,depth_image,yl[ind],xl[ind])
            if abs(TablePlane.a * x + TablePlane.b * y + TablePlane.c * z + TablePlane.d) / (TablePlane.a ** 2 + TablePlane.b**2 + TablePlane.c**2)**0.5 < 0.008:
                old_xyr.append([xl[ind],yl[ind],rl[ind]])

        Pt2 = []
        Pt3 = []
        Pt4 = []
        Color4 = []

        for xyr in old_xyr:
            x2,y2,z2 = rgbdTools.getPosition(RealSense,depth_image,xyr[1],xyr[0])
            Pt2.append([x2,y2,z2])
            p_l = keyPoints.pointInRadius(xyr[0],xyr[1], 2)
            for p in p_l:
                m2,n2 = p
                x,y,z = rgbdTools.getPosition(RealSense,depth_image,m2,n2)
                Pt3.append([x,y,z])

        Point2.points = o3d.utility.Vector3dVector(np.array(Pt2))
        Point3.points = o3d.utility.Vector3dVector(np.array(Pt3))
        FeatureList,new_xyr = registration.extractFeatures(Point2,old_xyr,n = 3)

        Trans_init,ind_l = registration.fastMatch(FeatureList,temFeatureList)

        for mm in range(0,480):
            for nn in range(0,640):
                x,y,z = rgbdTools.getPosition(RealSense,depth_image,mm,nn)
                if  y < (TablePlane.a * x + TablePlane.c * z + TablePlane.d)/(-TablePlane.b) -0.007 and z > 0.15 and z < 0.4:
                    Pt4.append([x,y,z])
                    Color4.append(rgbdTools.getColor(color_image,mm,nn))

        Point4.points = o3d.utility.Vector3dVector(np.array(Pt4))
        Point4.colors = o3d.utility.Vector3dVector(np.array(Color4))

        Point4 = Point4.voxel_down_sample(voxel_size=0.001)
        cl, ind = Point4.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.0)
        Point4 = Point4.select_down_sample(ind)

        globalPointcloud2 += Point4.transform(Trans_init)

        print("bottom view_{} is added.".format(bottom_i))

    # globalPointcloud2 = globalPointcloud2.voxel_down_sample(voxel_size=0.001)
    # cl, ind = globalPointcloud2.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.0)
    # globalPointcloud2 = globalPointcloud2.select_down_sample(ind)

    o3d.io.write_point_cloud("./output/bottom.pcd",globalPointcloud2)
    o3d.io.write_point_cloud("./output/bottom.ply",globalPointcloud2)

    print("bottom part is generated!")

    trans = np.array(
        [[-0.28131547, -0.95610337 ,-0.08202406, -0.03595882],
        [ 0.91153186 ,-0.29295931 , 0.28859056 ,-0.03853821],
        [-0.29995212 , 0.00641745 , 0.95393267 , 0.00908461],
        [ 0.        ,  0.        ,  0.         , 1.        ]]
    )

    globalPointcloud = globalPointcloud2 + globalPointcloud1.transform(trans)

    o3d.io.write_point_cloud("./output/dog.pcd",globalPointcloud)
    o3d.io.write_point_cloud("./output/dog.ply",globalPointcloud)

    print("global registration is done.")




