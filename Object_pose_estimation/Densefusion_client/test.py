import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
import open3d as o3d
import os
from torchvision import transforms
import socket 
import copy

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


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
    s.connect(('titanxp.sure-to.win',8899))
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
    vis.create_window("Pose in virtual environment",640,480)
    ctr = vis.get_view_control()
    camParam = o3d.camera.PinholeCameraParameters()
    camParam.intrinsic = o3d.camera.PinholeCameraIntrinsic(640 ,480, 616.8676, 617.0631 ,319.5 ,239.5)
    # vis.create_window("Pointcloud")
    pointcloud = o3d.geometry.PointCloud()
    i = 0

    objlist = [1,2,3]
    colorlist = [(0,255,0),(0,0,255),(255,0,0)]

    pcd1 = o3d.io.read_point_cloud('models/obj_{:02d}.pcd'.format(1)).transform(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    pcd2 = o3d.io.read_point_cloud('models/obj_{:02d}.pcd'.format(2)).transform(np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]))
    pcd3 = o3d.io.read_point_cloud('models/obj_{:02d}.pcd'.format(3)).transform(np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]))

    pcd = []
    pcd.append(pcd1)
    pcd.append(pcd2)
    pcd.append(pcd3)

    while True:
        dt0 = datetime.now()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        color_image2 = copy.deepcopy(color_image1)

        # color_image = np.uint8(np.clip((0.9 * color_image + 10), 0, 255))
        # color_image = cv2.GaussianBlur(color_image, ksize=(3, 3), sigmaX=0, sigmaY=0)

        # color_image = cv2.GaussianBlur(color_image, (5, 5), 0)

        imgg = cv2.imencode('.jpg', color_image)[1]
        data_encode = np.array(imgg)

        str_encode = data_encode.tostring()
        # length = b'%d'%(len(str_encode))
        # length = str.encode(str(len(str_encode)).ljust(16))
        length = str.encode(str(len(str_encode)).ljust(16))
        # print(len(str_encode))
        s.send(length)
        s.send(str_encode)

        dimgg = cv2.imencode('.png', depth_image)[1]
        data_encode = np.array(dimgg)

        str_encode = data_encode.tostring()
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

        len_mat= recvall(s,16)
        string_mat = recvall(s, int(len_mat))     

        final_mat = np.fromstring(string_mat).reshape(3,4,4)
        print(len(final_mat))



        print()
        print(final_mat[:,:,:])
        print()

        # final_mat[:3,3] *= 1e0

############################# bounding  box
        
        # arr = copy.deepcopy(labels)

        # arr = np.where(arr!=obj  , 0, arr )
        # arr = np.where(arr ==obj , 255, arr)

        # contours, hierarchy = cv2.findContours(arr,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # contour = 0
        # x, y, w, h = 0,0,0,0
        # for i in range(len(contours)):
        #     area = cv2.contourArea(contours[i])
        #     if area > 2000:
        #         contour = contours[i]
        #         x, y, w, h = cv2.boundingRect(contour) 
        #         # print(area)


        semantic_color = cv2.applyColorMap(labels*22,cv2.COLORMAP_HSV)
        semantic_RGB = cv2.cvtColor(semantic_color, cv2.COLOR_BGR2RGB)

        # bbox = [x,y,w,h]
        # # rmin,rmax,cmin,cmax = get_bbox(bbox)

        # cv2.rectangle(color_image1, (x, y), (x+w, y+h), (0,0,255), 1)
##################################################################################

        cv2.imshow('Color Stream', color_image1)
        cv2.imshow('Segmentation Stream', semantic_color )

        camParam.extrinsic = final_mat[2,:,:]
        pcd1_ = copy.deepcopy(pcd1)
        pcd1_ = pcd1_.transform(np.linalg.inv(final_mat[2,:,:]).dot(final_mat[0,:,:]))
        pcd2_ = copy.deepcopy(pcd2)
        pcd2_ = pcd2_.transform(np.linalg.inv(final_mat[2,:,:]).dot(final_mat[1,:,:]))
        pcd3_ = copy.deepcopy(pcd3)
        # pcd3_ = pcd3_.transform(np.linalg.inv(final_mat[2,:,:]).dot(final_mat[2,:,:]))
        # pcd.transform(final_mat)   
        # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        pointcloud.clear()
        pointcloud += (pcd1_ + pcd2_ + pcd3_)

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True

        ctr.convert_from_pinhole_camera_parameters(camParam)

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
            cv2.imwrite('./'+str(i)+'.png',semantic_color )
            cv2.imwrite('./output/depth_'+str(i)+'.png',depth_image)
            cv2.imwrite('./output/color_'+str(i)+'.png',color_image2)
            # o3d.io.write_point_cloud('./output/pointcloud_'+str(i)+'.pcd', pcd)
            print('No.'+str(i)+' shot is saved.' )
            i += 1


        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            vis.destroy_window()
            s.close()
            break


    pipeline.stop()

