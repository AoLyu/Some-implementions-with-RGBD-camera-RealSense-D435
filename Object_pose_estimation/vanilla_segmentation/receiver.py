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

sys.path.append("..")
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--model_save_path', default='trained_models/', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--resume_model', default='model_current.pth', help="resume model name")
opt = parser.parse_args()

model = segnet()
model = model.cuda()

if opt.resume_model != '':
    checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
    model.load_state_dict(checkpoint)
elif opt.resume_model == '':
    print('please direct the trained model!')

rgb_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

address = ('titanx.sure-to.win', 8899)
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
        # print(type(length))
        # print(length)
        stringData = recvall(sock,int(length))
        if not stringData:
            break
        data = np.fromstring(stringData,np.uint8)
        # sock.send(str(time.time()).encode()) 

        # # print('data length:',len(data))
        color_image = cv2.imdecode(data,cv2.IMREAD_COLOR)


        color_image = np.asanyarray(color_image)

        rgb = color_image.astype(np.float32)
        rgb = torch.from_numpy(rgb).cuda().permute(2, 0, 1).contiguous()
        rgb = rgb_norm(rgb).view(1,3,480,640)
        semantic = model(rgb)
        semantic = semantic.view(11,480,640).permute(1,2,0).contiguous()
        max_values , labels = torch.max( semantic , 2 )
        labels = labels.cpu().detach().numpy().astype(np.uint8)

        print(np.unique(labels))

        semantic_color = cv2.applyColorMap(labels*20,cv2.COLORMAP_HSV)

        encode_labels = cv2.imencode('.jpg',labels)[1]

        label_encode = np.array(encode_labels)

        str_label = label_encode.tostring()

        label_length = str.encode(str(len(str_label)).ljust(16))
        
        sock.send(label_length)
        sock.send(str_label)
        # flag+=1
    # cv2.destroyWindow('color label')
    sock.close()
    print('connection from %s:%s  is closed'% addr)

while 1:
    sock, addr = s.accept()
    t = threading.Thread(target = tcplink,args = (sock,addr))
    t.start()

s.close()
