import socket
import cv2
import numpy as np
import time
import os
import random
import argparse
import time
from PIL import Image
import numpy as np
import torch
import numpy.ma as ma
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
from segnet import SegNet as segnet
from PIL import ImageEnhance
from PIL import ImageFilter
import sys
import socket
import threading

from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
# from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import torchvision.transforms as transforms
import torchvision.utils as vutils
sys.path.append("..")
from lib.utils import setup_logger
from lib.network import PoseNet, PoseRefineNet
import copy
import json

def load_json(path, keys_to_int=False):
    def convert_keys_to_int(x):
        return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

    with open(path, 'r') as f:
        if keys_to_int:
            content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
        else:
            content = json.load(f)

    return content

def get_bbox(bbx):
    # bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
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

parser = argparse.ArgumentParser()
parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--model_save_path', default='trained_models/', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--resume_model', default='eval_RT/models/seg_model.pth', help="resume model name")
parser.add_argument('--model', type=str, default = 'eval_RT/models/pose_model.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = 'eval_RT/models/pose_refine_model.pth',  help='resume PoseRefineNet model')
opt = parser.parse_args()

model = segnet()
model = model.cuda()

if opt.resume_model != '':
    checkpoint = torch.load('{}'.format(opt.resume_model))
    model.load_state_dict(checkpoint)
elif opt.resume_model == '':
    print('please direct the trained model!')

rgb_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

cam_cx = 319.5701
cam_cy = 233.0649
cam_fx = 616.8676
cam_fy = 617.0631

cam_scale = 1000.0
num_obj = 3
img_width = 480
img_length = 640
num_points = 700
iteration = 3
bs = 1

# knn = KNearestNeighbor(1)
estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

###################   address  ##########################
address = ('titanxp.sure-to.win', 8899)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)
s.listen(True)
print('waiting for connection...')


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def tcplink(sock,addr):
    print("Accept a new connection from %s:%s..."%addr)
    sock.send(b'Welcome!')
    # flag = 1
    # cv2.namedWindow('color label')
    while True:
        length = recvall(sock,16)
        if not length:
            break
        stringData = recvall(sock,int(length))
        if not stringData:
            break
        data = np.fromstring(stringData,dtype = 'uint8')

        color_image = cv2.imdecode(data,cv2.IMREAD_COLOR)
        color_image = np.asanyarray(color_image)

############ depth image 
        length2 = recvall(sock,16)
        if not length2:
            break
        stringData2 = recvall(sock,int(length2))
        if not stringData:
            break
        data2 = np.fromstring(stringData2,dtype = 'uint8')
        depth_image = cv2.imdecode(data2,-1) 
        depth_image = np.asanyarray(depth_image)

        rgb2 =copy.deepcopy(color_image)
        rgb3 = Image.fromarray(rgb2.astype('uint8')).convert('RGB')
        rgb3 = ImageEnhance.Brightness(rgb3).enhance(1.4)
        # rgb3 = ImageEnhance.Contrast(rgb3).enhance(1.4)
        # rgb3 = rgb3.filter(ImageFilter.GaussianBlur(radius=2))
        # rgb3 = copy.deepcopy(rgb2)

        rgb = np.array(rgb2).astype(np.float32)

        rgb = torch.from_numpy(rgb).cuda().permute(2, 0, 1).contiguous()
        rgb = rgb_norm(rgb).view(1,3,480,640)
        semantic = model(rgb)
        semantic = semantic.view(4,480,640).permute(1,2,0).contiguous()
        max_values , labels = torch.max( semantic , 2 )
        labels = labels.cpu().detach().numpy().astype(np.uint8)

        encode_labels = cv2.imencode('.jpg',labels)[1]
        # cv2.waitKey()
        label_encode = np.array(encode_labels)
        str_label = label_encode.tostring()

        label_length = str.encode(str(len(str_label)).ljust(16))
        sock.send(label_length)
        sock.send(str_label)
        ######## pose prediction
        obj_ids = np.unique(labels)[1:]
        print(obj_ids)

        posenetlist = [1,2,3]
        zero_mat = np.zeros((4,4))
        pose_result = []

        for obj in posenetlist:
            arr = copy.deepcopy(labels)
            arr = np.where(arr != obj,   0, arr)
            arr = np.where(arr == obj, 255, arr)
            
            contours,hierachy = cv2.findContours(arr,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            contour = 0
            x,y,w,h = 0,0,0,0

            if len(contours)==0:
                continue
                
            continue_flag = 0

            for i in range(len(contours)):
                area =cv2.contourArea(contours[i])
                if area > 2500:
                    contour =contours[i]
                    x,y,w,h =cv2.boundingRect(contour)
                    continue_flag = 0
                    break
                else:
                    continue_flag = 1

            if (continue_flag==1):
                pose_result.append(zero_mat)
                continue
            idx = posenetlist.index(obj)

            bbx = []

            bbx.append(y)
            bbx.append(y+h)
            bbx.append(x)
            bbx.append(x+w)

            rmin, rmax, cmin, cmax = get_bbox(bbx)

            # img  = copy.deepcopy(color_image)

            img_masked = np.transpose(np.array(rgb3)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

            img_masked_shape = img_masked.shape

            mask_label = ma.getmaskarray(ma.masked_equal(labels, np.array(obj)))

            choose = mask_label[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            choose_shape = choose.shape

            xmap = np.array([[j for i in range(640)] for j in range(480)])
            ymap = np.array([[i for i in range(640)] for j in range(480)])

            depth = copy.deepcopy(depth_image)
                    
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cam_scale = 1.0
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            cloud /= 1000
            cloud_shape = cloud.shape
            # points = cloud.reshape((1,cloud_shape[0,cloud_shape[1]]))
            # print(cloud_shape)
            
            # points = cloud.view(1,cloud_shape[0],cloud_shape[1])
            # choose = choose.view(1,choose_shape[0],choose[1])
            # img_masked = img_masked.reshape((1,img_masked_shape[0],img_masked_shape[1],img_masked_shape[2]))

            if cloud.shape[0] < 2:
                print('Lost detection!')
            # fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
            # continue
            points = torch.from_numpy(cloud.astype(np.float32)).cuda()
            choose = torch.LongTensor(choose.astype(np.int32)).cuda()
            img = rgb_norm(torch.from_numpy(img_masked.astype(np.float32))).cuda()
            idx = torch.LongTensor([idx]).cuda()

            points = points.view(1,cloud_shape[0],cloud_shape[1]).contiguous()
            img = img.view(1,img_masked_shape[0],img_masked_shape[1],img_masked_shape[2]).contiguous()

            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)

            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t

                new_points = torch.bmm((points - T), R).contiguous()
                pred_r, pred_t = refiner(new_points, emb, idx)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)
                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final

            my_mat_final[:3,:3] = quaternion_matrix(my_r)[:3, :3] 
            my_mat_final[:3,3] = my_t
            # pose_mat[obj-1,3,:] = np.array([0,0,0,1])
            # pose_mat[:,:,obj-1]=my_mat_final
            pose_result.append(my_mat_final)
            if (obj == posenetlist[-1]):
                break

        pose_result = np.array(pose_result)
        print(pose_result)
        my_mat_str = pose_result.tostring()
        length = str.encode(str(len(my_mat_str)).ljust(16))

        sock.send(length)
        sock.send(my_mat_str)
        print()
        print(my_mat_final)
        print()
    sock.close()
    print('connection from %s:%s  is closed'% addr)

while 1:
    sock, addr = s.accept()
    t = threading.Thread(target = tcplink,args = (sock,addr))
    t.start()

s.close()
