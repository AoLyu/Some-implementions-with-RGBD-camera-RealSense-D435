from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json
import open3d
import h5py


class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 1024, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.obj_list = [1,2,3,6,7,8,9,10,11]
        # self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        # self.cat = {}
        self.data_info = []
        if train:
            for obj in self.obj_list:
                for jj in range(7600):
                    self.data_info.append([obj,jj])
        else:
            for obj in self.obj_list:
                for jj in range(7600,8000):
                    self.data_info.append([obj,jj])
        self.classes = list(range(11))




        #print(self.num_seg_classes)


    def __getitem__(self, index):
        # fn = self.datapath[index]
        obj_id = self.data_info[index][0]
        jj =self.data_info[index][1]

        data = h5py.File(self.root+'{}.h5'.format(obj_id),'r')
        point_set2048 = np.array(data['data'][:][jj])
        point_set = point_set2048[::2,:]
        # seg = np.array(data['label'][:][jj])
        data.close()

        if len(point_set) > self.npoints:
            c_mask = np.zeros(len(point_set), dtype=int)
            c_mask[:self.npoints] = 1
            np.random.shuffle(c_mask)
            choose = np.array(range(len(point_set)))
            choose = choose[c_mask.nonzero()]
            point_set = point_set[choose, :]
            # choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        elif len(point_set) < self.npoints:
            choose = np.array(range(len(point_set)))
            choose = np.pad(choose, (0, self.npoints - len(choose)), 'wrap')
            point_set = point_set[choose, :]


        ind = self.obj_list.index(obj_id)   
        # point_set = point_set[choose, :]        
        # #resample
        # point_set = point_set[choice, :]
        # print(point_set.shape)
        point_set = torch.from_numpy(point_set.astype(np.float32))
        # seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([ind]).astype(np.int64))
        # if self.classification:
        return point_set, cls
        # else:
        #     return point_set, seg

    def __len__(self):
        return len(self.data_info)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = '/data2/leo/bop/bop_renderer/samples/cls/', class_choice = ['Chair'],classification =True)
    print(len(d))
    point_set, cls = d[0]
    print('ps')
    print(point_set.size())
    print(cls.size())
    # print(np.unique(np.asanyarray(seg)))
    # print('seg:',seg)
    # print(ps.size(), ps.type(), seg.size(),seg.type())

    # d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    # print(len(d))
    # ps, cls = d[0]
    # print(ps.size(), ps.type(), cls.size(),cls.type())
