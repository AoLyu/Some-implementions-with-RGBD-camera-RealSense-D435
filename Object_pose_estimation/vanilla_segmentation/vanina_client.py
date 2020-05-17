import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
import open3d as o3d
import os
from torchvision import transforms
import socket 


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
    # trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect(('titanx.sure-to.win',8899))
    print(s.recv(1024).decode('utf-8'))

    align = rs.align(rs.stream.color)
    #align = rs.align(rs.stream.depth)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # print(type(pinhole_camera_intrinsic))
    
    cv2.namedWindow('Color Stream', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Segmentation Stream', cv2.WINDOW_AUTOSIZE)

    geometrie_added = False
    vis = o3d.visualization.Visualizer()
    #vis.create_window("Pointcloud",640,480)
    vis.create_window("Pointcloud")
    pointcloud = o3d.geometry.PointCloud()
    i = 0

    while True:
        dt0 = datetime.now()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)


        imgg = cv2.imencode('.jpg', color_image)[1]
        data_encode = np.array(imgg)

        str_encode = data_encode.tostring()
        # length = b'%d'%(len(str_encode))
        # length = str.encode(str(len(str_encode)).ljust(16))
        length = str.encode(str(len(str_encode)).ljust(16))
        # print(len(str_encode))
        s.send(length)
        s.send(str_encode)


######## receiving data

        len_label= recvall(s,16)
        stringlabel = recvall(s, int(len_label))

        labels = np.fromstring(stringlabel, dtype='uint8')
        labels =cv2.imdecode(labels,0)
        labels  = np.asanyarray(labels)

#############################
        # print(np.unique(labels))
        semantic_color = cv2.applyColorMap(labels*20,cv2.COLORMAP_HSV)
        semantic_RGB = cv2.cvtColor(semantic_color, cv2.COLOR_BGR2RGB)
###############################

        # img = np.zeros((480,640),np.uint8)

        # img = np.where(labels == 6 , 255 , img)

        # contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        # contour = 0
        # for i in range(len(contours)):
        #     area = cv2.contourArea(contours[i])
        #     if area > 300:
        #         contour = contours[i]
        #         x, y, w, h = cv2.boundingRect(contour) 

        # # print(contour.squeeze())
        # cv2.drawContours(img,contour, 0,255,1)
        # cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)


        cv2.imshow('Color Stream', color_image1)
        cv2.imshow('Segmentation Stream', semantic_color )
        # cv2.imshow('Segmentation2 Stream', sementic_RGB )


        depth = o3d.geometry.Image(depth_image)
        color = o3d.geometry.Image(np.asarray(semantic_RGB))
        # color = Image(np.asarray(semantic))
        # color = Image(color_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

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
        process_time = datetime.now() - dt0
        print("FPS = {0}".format(int(1/process_time.total_seconds())))


        key = cv2.waitKey(1)

        if key & 0xFF == ord('s'):
            if not os.path.exists('./output/'): 
                os.makedirs('./output')
            # cv2.imwrite('./output/label_'+str(i)+'.png',semantic_color)
            cv2.imwrite('./output/depth_'+str(i)+'.png',semantic_color)
            cv2.imwrite('./output/color_'+str(i)+'.png',color_image1)
            o3d.io.write_point_cloud('./output/pointcloud_'+str(i)+'.pcd', pcd)
            print('No.'+str(i)+' shot is saved.' )
            i += 1


        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            vis.destroy_window()
            s.close()
            break


    pipeline.stop()

