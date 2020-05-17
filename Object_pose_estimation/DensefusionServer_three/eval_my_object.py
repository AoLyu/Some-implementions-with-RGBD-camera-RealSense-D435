import _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.my_object.dataset import PoseDataset as PoseDataset_my_object
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

num_objects = 11
objlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
num_points = 1000
iteration = 2
dataset_config_dir = 'datasets/my_object/dataset_config'
output_result_dir = 'experiments/eval_result/my_object'


estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

testdataset = PoseDataset_my_object('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diam = [103.3035,132.6247,128.2806,102.6255,129.7793,108.1525,103.8450,82.2438,109.7894,134.5150,107.5186]
diameter = []
for d in diam:
    diameter.append(d / 1000.0 * 0.1)
# meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
# meta = yaml.load(meta_file)
# for obj in objlist:
#     diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx = data
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(target).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()
    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, 0.0, False)
    for ite in range(0, iteration):
        pred_r, pred_t = refiner(new_points, emb, idx)
        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

    if dis.item() < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis.item()))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis.item()))
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis.item()))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis.item()))
    num_count[idx[0].item()] += 1

for i in range(num_objects):
    print('Object {0} success rate: {1}'.format(objlist[i], success_count[i] / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], success_count[i] / num_count[i]))
print('ALL success rate: {0}'.format(sum(success_count) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(sum(success_count) / sum(num_count)))
fw.close()
