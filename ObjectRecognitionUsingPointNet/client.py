import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
from open3d import *
import os
from tools import rgbdTools
from scipy import optimize
from tools import keyPoints
import socket
import copy


def res(p,xarray,yarray,zarray):
    a,b,c,d = p
    return abs(a*xarray+b*yarray+c*zarray+d)/(a**2+b**2+c**2)**0.5

def maxl(lis):
    max_i = lis[0]
    for ii in lis:
        if ii > max_i:
            max_i = ii
    return max_i

def minl(lis):
    min_i = lis[0]
    for ii in lis:
        if ii < min_i:
            min_i = ii
    return min_i

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

if __name__=="__main__":

    npoints = 1024
    obj_list = [1,2,3,6,7,8,9,10,11]
    obj_list = ['backpack','polar bear','box','duck','turtle','whale','dog','elephant','horse']
    color_list = [[96/255,96/255,96/255],[1,97/255,0],[227/255,207/255,87/255],[176/255,224/255,230/255],
                [106/255,90/255,205/255],[56/255,94/255,15/255],[61/255,89/255,171/255],[51/255,161/255,201/255],
                [178/255,34/255,34/255],[138/255,43/255,226/255]]
    


    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect(('titanxp.sure-to.win',8899))
    print(s.recv(1024).decode('utf-8'))

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
    pinhole_camera_intrinsic = PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # print(type(pinhole_camera_intrinsic))
    
    cv2.namedWindow('Color Stream', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE)

    cam = rgbdTools.Camera(616.8676147460938,617.0631103515625,319.57012939453125,233.06488037109375)


    geometrie_added = False
    vis = Visualizer()
    #vis.create_window("Pointcloud",640,480)
    vis.create_window("Pointcloud")
    pointcloud = PointCloud()
    i = 0
    plane_flag = 0

    tablePlane = keyPoints.Plane()

    z_min = None
    z_max = None
    x_min = None
    x_max = None

    while True:
        dt0 = datetime.now()
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


        if plane_flag == 0:
            chessboard_found1, corners1 = cv2.findChessboardCorners(color_image1, (9, 6))
            corners = np.asanyarray(corners1).squeeze()
            if chessboard_found1:
                # cv2.drawChessboardCorners(color_image1,(9,6),corners1,chessboard_found1)
                Points = []
                for corner in corners:
                    n = int(round(corner[0]))
                    m = int(round(corner[1]))
                    if depth_image[m,n] > 0 :
                        x,y,z = rgbdTools.getPosition(cam,depth_image,m,n)
                        Points.append([x,y,z])
                Points = np.array(Points)
                a = (Points[1,1] - Points[0,1])*(Points[2,2] - Points[0,2]) - (Points[1,2] - Points[0,2])*(Points[2,1] - Points[0,1])
                b = (Points[1,2] - Points[0,2])*(Points[2,0] - Points[0,0]) - (Points[1,0] - Points[0,0])*(Points[2,2] - Points[0,2])
                c = (Points[1,0] - Points[0,0])*(Points[2,1] - Points[0,1]) - (Points[1,1] - Points[0,1])*(Points[2,0] - Points[0,0])
                d = 0 - (a * Points[0,0] + b*Points[0,1] + c*Points[0,2])

                z_min = minl(Points[:,2])
                z_max = maxl(Points[:,2])
                x_min = minl(Points[:,0])
                x_max = maxl(Points[:,0])

                r = optimize.leastsq(res,[a,b,c,d],args=(Points[:,0],Points[:,1],Points[:,2]))
                a,b,c,d = r[0]
                tablePlane.getParam(a,b,c,d)


                plane_flag = 1
                print('plane coefficient is calculated.')
                print("press 'a' to recalculate the plane coefficient")
            else:
                print('please put an chessboard in the view')

        Pt = []
        obj_pt = []
        colors =[] 
        pcd = open3d.geometry.PointCloud()
        obj_pcd = open3d.geometry.PointCloud()
        pre_obj_index = 0

        # print(x_max, x_min , z_max, z_min)

        if plane_flag ==1:
            obj_color_ind_list = []
            obj_color_ind = 0
            for mm in range(0,480,4):
                for nn in range(0,640,4):
                    if depth_image[mm,nn] > 100 and depth_image[mm,nn] < 500 :
                        x,y,z = rgbdTools.getPosition(cam,depth_image,mm,nn)
                        if  y > (tablePlane.a * x + tablePlane.c * z + tablePlane.d)/(-tablePlane.b) - 0.01 and y < (tablePlane.a * x + tablePlane.c * z + tablePlane.d)/(-tablePlane.b) + 0.01:
                            # label_l.append(1)
                            Pt.append([x,y,z])
                            colors.append([0,1.0,0])
                        elif y < (tablePlane.a * x + tablePlane.c * z + tablePlane.d)/(-tablePlane.b) - 0.01 :
                            # colors.append([1.0,0,0])
                            obj_pt.append([x,y,z])
                        else:
                            Pt.append([x,y,z])
                            colors.append([96/255,96/255,96/255])
            label_index = ''
            obj_pt2 = copy.deepcopy(obj_pt)
            if len(obj_pt2) > 0:
                obj_pcd.points =  open3d.utility.Vector3dVector(np.array(obj_pt2))
                obj_pcd.paint_uniform_color([1.0,0,0])

            if len(obj_pt) > 0:
                obj_pt = np.array(obj_pt)
                # print(obj_pt.shape)
                mean_point = np.mean(obj_pt,axis=0)
                obj_pt = obj_pt-mean_point

                if len(obj_pt) > npoints:
                    c_mask = np.zeros(len(obj_pt),dtype=int)
                    c_mask[:npoints] = 1
                    np.random.shuffle(c_mask)
                    choose = np.array(range(len(obj_pt)))
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.array(range(len(obj_pt)))
                    choose = np.pad(choose,(0,npoints-len(choose)),'wrap')
                point_set = obj_pt[choose,:]
                # print(point_set.shape)
                str_encode = point_set.tostring()
                length = str.encode(str(len(str_encode)).ljust(16))
                s.send(length)
                s.send(str_encode)
                label_index = recvall(s,2)
                print('the object is',obj_list[int(label_index)])
            
            if len(obj_pt2) >0:
                print(len(obj_pt2))
                obj_pcd.paint_uniform_color(color_list[int(label_index)+1])

            pcd.points = open3d.utility.Vector3dVector(np.array(Pt))
            pcd.colors = open3d.utility.Vector3dVector(np.array(colors))

        else:                
            depth = Image(depth_image)
            color = Image(color_image)
            rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
            pcd = create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

             
        # print('obj_num:',obj_num)

        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        cv2.imshow('Color Stream', color_image1)
        cv2.imshow('Depth Stream', depth_color_image )




        if not pcd:
            continue

        pointcloud.clear()

        # print('obj_points',len(obj_pcd.points))

        pointcloud += obj_pcd
        pointcloud += pcd

            
        pointcloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        # pcd = voxel_down_sample(pcd, voxel_size = 0.003)

        # if len(obj_pt) > 0:
        # pointcloud += obj_pcd

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True


        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        process_time = datetime.now() - dt0
        print("FPS = {0}".format(int(1/process_time.total_seconds())))


        key = cv2.waitKey(1)

        if key & 0xFF == ord('a'):
            plane_flag = 0

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            vis.destroy_window()

            break


    pipeline.stop()

