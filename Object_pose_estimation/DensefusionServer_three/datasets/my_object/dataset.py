import torch.utils.data as data
from PIL import Image
from PIL import ImageFilter
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
from PIL import ImageEnhance
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import random


class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        self.objlist = [1,2,3,4]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.list_meta = []
        self.pt = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        # item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                meta = load_json('/data/leo/render_drink_box/{:06d}/scene_gt.json'.format(item))
                for image_id in range(10000):
                    self.list_rgb.append('/data/leo/render_drink_box/{:06d}/rgb/{:06d}.png'.format(item,image_id))
                    self.list_depth.append('/data/leo/render_drink_box/{:06d}/depth/{:06d}.png'.format(item,image_id))
                    self.list_meta.append(meta['{}'.format(image_id)][0])
                    self.list_obj.append(item)
            # elif self.mode == 'test':
            else:
                meta = load_json('/data/leo/render_drink_box/{:06d}/scene_gt.json'.format(item))
                for image_id in range(10000,10500):
                    self.list_rgb.append('/data/leo/render_drink_box/{:06d}/rgb/{:06d}.png'.format(item,image_id))
                    self.list_depth.append('/data/leo/render_drink_box/{:06d}/depth/{:06d}.png'.format(item,image_id))
                    self.list_meta.append(meta['{}'.format(image_id)][0])
                    self.list_obj.append(item)
            self.pt[item] = ply_vtx('/data/leo/render_drink_box/obj_{:06d}.ply'.format(item))
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 319.5701
        self.cam_cy = 233.0649
        self.cam_fx = 616.8676
        self.cam_fy = 617.0631

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 2700
        self.num_pt_mesh_small = 1700
        self.symmetry_obj_idx = []

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        random_choice = random.randint(0,100)
        color_ran = random.randint(8,17)
        img = ImageEnhance.Brightness(img).enhance(color_ran * 0.1)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        # ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        # label = np.array(Image.open(self.list_label[index]))

        # rand_back = random.randint(0,34)

        # background  = np.array(Image.open("/data2/leo/bop/background1/{:02d}.png".format(rand_back)))

        ys,xs = np.nonzero(depth > 0)

        obj_bb = [xs.min(),ys.min(),xs.max()-xs.min(), ys.max()-ys.min() ]

        rmin, rmax, cmin, cmax = get_bbox(obj_bb)

        obj = self.list_obj[index]

        meta = self.list_meta[index]

        mask_label = ma.getmaskarray(ma.masked_not_equal(depth, np.array(0)))

        mask = mask_label

        if self.add_noise:
            img = self.trancolor(img)
        # ys,xs = np.nonzero(depth > 0)
        obj_bb = [xs.min(),ys.min(),xs.max()-xs.min(), ys.max()-ys.min() ]

        rmin, rmax, cmin, cmax = get_bbox(obj_bb)

        img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c'])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        if self.add_noise:
            cloud = np.add(cloud, add_t)

        #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        #for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        model_points = self.pt[obj] / 1000.0
        # print(len(model_points))
        # print(len(model_points))
        # print(len(model_points))
        dellist = [j for j in range(0, len(model_points))]
        if self.refine:
            dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)

        #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
        #for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0

        #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        #for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



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


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

def load_json(path, keys_to_int=False):
  """Loads content of a JSON file.

  :param path: Path to the JSON file.
  :return: Content of the loaded JSON file.
  """
  # Keys to integers.
  def convert_keys_to_int(x):
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

  with open(path, 'r') as f:
    if keys_to_int:
      content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
    else:
      content = json.load(f)

  return content
