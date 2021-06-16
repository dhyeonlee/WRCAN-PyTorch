import os

from data import common
from data import util

import numpy as np
import cv2

import torch
import torch.utils.data as data

class REDS_TEST(data.Dataset):
    def __init__(self, args, name='REDS_TEST', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.dir_lr = os.path.join(args.dir_data, 'REDS', 'test/test_blur_jpeg')
        self.filelist = sorted(util._get_paths_from_images(self.dir_lr))
        # for f in os.listdir(args.dir_demo):
        #     if f.find('.png') >= 0 or f.find('.jp') >= 0:
        #         self.filelist.append(os.path.join(args.dir_demo, f))
        # self.filelist.sort()


    def __getitem__(self, idx):
        # filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        filename, _ = os.path.splitext('/'.join(self.filelist[idx].split('/')[-2:]))

        lr = cv2.imread(self.filelist[idx], cv2.IMREAD_COLOR)[...,::-1]
 
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

        return lr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

