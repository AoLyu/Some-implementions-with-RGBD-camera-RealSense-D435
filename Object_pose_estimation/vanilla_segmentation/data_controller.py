import torch
import numpy as np
from PIL import Image
import numpy.ma as ma
import torch.utils.data as data
import copy
from torchvision import transforms
import scipy.io as scio
import torchvision.datasets as dset
import random
import scipy.misc
import scipy.io as scio
import os
from PIL import ImageEnhance
from PIL import ImageFilter

class SegDataset(data.Dataset):
    def __init__(self, root_dir, mode, use_noise, num=1000):
        self.train_rgb_path = []
        self.train_label_path = []

        self.test_rgb_path = []
        self.test_label_path = []

        # self.real_path = []
        self.use_noise = use_noise
        self.num = num
        self.root = root_dir
        self.mode = mode
        self.label_path = ''
        # input_file = open(txtlist)

        if self.mode == 'train':
            for image_id in range(20000):
                self.train_rgb_path.append('/data/leo/new_objseg_dataset/fusion/{:05d}.png'.format(image_id))
                self.train_label_path.append('/data/leo/new_objseg_dataset/label/{:05d}.png'.format(image_id))
            self.length = len(self.train_rgb_path)
        else:
            for image_id in range(19000,20000):
                self.test_rgb_path.append('/data/leo/new_objseg_dataset/fusion/{:05d}.png'.format(image_id))
                self.test_label_path.append('/data/leo/new_objseg_dataset/label/{:05d}.png'.format(image_id))
            self.length = len(self.test_rgb_path)

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.back_front = np.array([[1 for i in range(640)] for j in range(480)])

    def __getitem__(self, index):
        if self.mode == 'train' :
        #     self.label_path = 'data/train/labels/'+ self.path[index][-5:]
            label = np.array(Image.open(self.train_label_path[index]))
        else:
        #     self.label_path = 'data/test/labels/'+ self.path[index][-5:]
            label = np.array(Image.open(self.test_label_path[index]))
        # meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.path[index]))

        # if not self.use_noise:
        #     rgb = np.array(Image.open('{0}/{1}-color.png'.format(self.root, self.path[index])).convert("RGB"))
        # else:
        #     if self.txtlist[-20:-15] == 'train' :
        #         rgb = np.array(self.trancolor(Image.open('{0}/train/synthesis/{1}.png'.format(self.root, self.path[index])).convert("RGB")))
        #     else :
        #         rgb = np.array(self.trancolor(Image.open('{0}/test/synthesis/{1}.png'.format(self.root, self.path[index])).convert("RGB")))

        # if True:
        # if self.path[index][:8] == 'data_syn':
        if self.mode == 'train' :
            rgb = Image.open(self.train_rgb_path[index]).convert("RGB")
            rndint = random.randint(0,100)
            if rndint < 70:
            	rand_color = random.randint(5,20)
            	rgb = ImageEnhance.Color(rgb).enhance(0.1*rand_color)
            rndint = random.randint(0,100)
            if rndint < 70:
            	rand_color = random.randint(5,20)
            	rgb = ImageEnhance.Contrast(rgb).enhance(0.1*rand_color)
            rndint = random.randint(0,100)
            if rndint < 70:
            	rand_color = random.randint(5,20)
            	rgb = ImageEnhance.Sharpness(rgb).enhance(0.1*rand_color)
            if rndint < 70:
            	rand_color = random.randint(5,20)
            	rgb = ImageEnhance.Brightness(rgb).enhance(0.1*rand_color)
            rgb =rgb.filter(ImageFilter.GaussianBlur(radius=0.9))
   #          rgb = np.array(rgb)
   #          # seed = random.randint(10, self.back_len - 10)
   #          back_id = random.randint(0,29)
   #          back = np.array(Image.open('/data2/leo/bop/background/{:02d}.png'.format(back_id)).convert("RGB"))
   #          # back_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.path[seed])))
   #          # mask = ma.getmaskarray(ma.masked_equal(label, 0))
			# # rgb = np.array(rgb)
   #          # rgb = back * mask + rgb
   #          # label = back_label * mask + label
   #          for mm in range(480):
   #              for nn in range(640):
   #                  if label[mm,nn] != 0: 
   #                      back[mm,nn] = rgb[mm,nn,:3]

   #          back = Image.fromarray(back.astype('uint8')).convert("RGB")
            # rgb = ImageEnhance.Sharpness(rgb).enhance(0.9).filter(ImageFilter.GaussianBlur(radius=0.6))
   #          rgb = np.array(rgb)
   #          rgb = np.transpose(rgb, (2, 0, 1))
   #          # rgb = np.transpose(rgb, (2, 0, 1))
   #          rgb = rgb + np.random.normal(loc=0.0, scale=5.0, size=rgb.shape)
   #          rgb = np.transpose(rgb, (1, 2, 0))
   #          #scipy.misc.imsave('embedding_final/rgb_{0}.png'.format(index), rgb)
   #          #scipy.misc.imsave('embedding_final/label_{0}.png'.format(index), label)
        else:
            rgb = Image.open(self.test_rgb_path[index]).convert("RGB")
        
        rgb = np.array(self.trancolor(rgb))
            # rgb = np.transpose(rgb, (2, 0, 1))
            # rgb = rgb + np.random.normal(loc=0.0, scale=5.0, size=rgb.shape)
            # rgb = back * mask + rgb
            # label = back_label * mask + label
            # rgb = np.transpose(rgb, (1, 2, 0))


            
        if self.use_noise:
            choice = random.randint(0, 3)
            if choice == 0:
                rgb = np.fliplr(rgb)
                label = np.fliplr(label)
            elif choice == 1:
                rgb = np.flipud(rgb)
                label = np.flipud(label)
            elif choice == 2:
                rgb = np.fliplr(rgb)
                rgb = np.flipud(rgb)
                label = np.fliplr(label)
                label = np.flipud(label)
                

        # obj = meta['cls_indexes'].flatten().astype(np.int32)

        # obj = np.append(obj, [0], axis=0)
        target = copy.deepcopy(label)

        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = self.norm(torch.from_numpy(rgb.astype(np.float32)))
        target = torch.from_numpy(target.astype(np.int64))

        return rgb, target


    def __len__(self):
        return self.length

