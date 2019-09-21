import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pointnet import PointNetCls
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
import h5py 
import socket
import threading



parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
parser.add_argument('--model', type=str, default = './cls/cls_model_1024.pth',  help='model path')

opt = parser.parse_args()

obj_list = [1,2,3,6,7,8,9,10,11]
num_classes = len(obj_list)

classifier = PointNetCls(k = num_classes, num_points = opt.num_points)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

classifier = classifier.cuda()
classifier = classifier.eval()

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
            
        point_set = np.fromstring(stringData)
        # print(point_set.shape)
        point_set = point_set.reshape(1,1024,3)
        point_set = torch.from_numpy(point_set.astype(np.float32))
        points = Variable(point_set)
        points = points.transpose(2,1)
        points = points.cuda()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        pred = pred_choice.cpu().detach()
        a = pred[0]
        a = int(a)
        str_pred = str.encode(str(a)).ljust(2)
        sock.send(str_pred)
        # print('the object is ',obj_list[a])
    sock.close()
    print('connection from %s:%s  is closed'% addr)


while 1:
    sock, addr = s.accept()
    t = threading.Thread(target = tcplink,args = (sock,addr))
    t.start()

s.close()
