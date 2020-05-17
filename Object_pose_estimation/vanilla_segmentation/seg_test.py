import os
import random
import argparse
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
from segnet import SegNet as segnet
import sys

import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

from datetime import datetime
from open3d import *


sys.path.append("..")
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--logs_path', default='logs/', help="path to save logs")
parser.add_argument('--model_save_path', default='trained_models/', help="path to save models")
parser.add_argument('--log_dir', default='logs/', help="path to save logs")
parser.add_argument('--resume_model', default='', help="resume model name")
parser.add_argument("--realsense_bag", default='object_detection.bag', help="Path to the realsense bag file")
opt = parser.parse_args()



model = segnet()
model = model.cuda()

if opt.resume_model != '':
    checkpoint = torch.load('{0}/{1}'.format(opt.model_save_path, opt.resume_model))
    model.load_state_dict(checkpoint)
elif opt.resume_model == '':
    print('please direct the trained model!')

rgb_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


if __name__ == '__main__':
    try:
    	color_image = cv2.imread('color_0.png')
    	color_image1 = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    	color_image1 = np.asanyarray(color_image1)
    	rgb = np.transpose(color_image1, (2, 0, 1))
    	rgb = rgb_norm(torch.from_numpy(rgb.astype(np.float32)))
    	rgb = rgb.reshape(1,3,480,640)
    	rgb = Variable(rgb).cuda()
    	semantic = model(rgb)
    	semantic = semantic.cpu().detach().numpy()
    	print(semantic.shape)
    	semantic = semantic.reshape(22,480,640)
    	semantic = np.transpose(semantic,(1,2,0))
    	semantic = np.argmax(semantic,axis=2)
    	semantic = semantic.astype(np.uint8)
    	print(semantic.shape)
    	cv2.imwrite('label-.png',semantic)
    	cv2.imshow('Color Stream', color_image)
    	#cv2.imshow('bgr Stream', color_image)
    	cv2.imshow('Mask Stream', semantic*20)
    	cv2.waitKey(0)
    	cv2.destroyAllWindows()
    finally:
    	pass


