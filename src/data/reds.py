# -*- coding: utf-8 -*-
# @Author: Donghyeon Lee (donghyeon1223@gmail.com)
# @Date:   2021-01-10 13:26:39
# @Last Modified by:   Donghyeon Lee (donghyeon1223@gmail.com)
# @Last Modified time: 2021-03-06 00:59:28
import os
from data import srdata
from data import util
from data import common

import pickle
import cv2
import numpy as np

class REDS(srdata.SRData):
    def __init__(self, args, name='REDS', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]


        self.begin, self.end = list(map(lambda x: int(x), data_range))

        # Contents below are in init() of SRData.
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        self.flag_ae_loss = True if (args.loss.lower().find('ae') >= 0 and train) else False


        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr, list_hr_ae = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr, self.images_hr_ae = list_hr, list_lr, list_hr_ae
        elif args.ext.find('sep') >= 0:
            os.makedirs(self.dir_hr.replace(self.apath, path_bin), exist_ok=True)
            os.makedirs(self.dir_lr.replace(self.apath, path_bin), exist_ok=True)
            if self.flag_ae_loss:
                os.makedirs(self.dir_hr_ae.replace(self.apath, path_bin), exist_ok=True)
            
            self.images_hr, self.images_lr, self.images_hr_ae = [], [[] for _ in self.scale], []
            
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) 
            for h_ae in list_hr_ae:
                b = h_ae.replace(self.apath, path_bin)
                b = b.replace(self.ext[2], '.pt')
                self.images_hr_ae.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 

                
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def _scan(self):
        names_hr = sorted(util._get_paths_from_images(self.dir_hr))
        names_lr = []
        names_lr.append(sorted(util._get_paths_from_images(self.dir_lr)))
        ## Set data range
        self.end = min(len(names_hr), self.end)
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        if self.flag_ae_loss:
            names_hr_ae = sorted(util._get_paths_from_images(self.dir_hr_ae))
            names_hr_ae = names_hr_ae[self.begin-1:self.end]
        else:
            names_hr_ae = []

        return names_hr, names_lr, names_hr_ae

    def _return_ksize(self):
        ksize_stidx = self.name.lower().find('ksize') + len('ksize')
        ksize_edidx = ksize_stidx
        while ksize_edidx < len(self.name) and ('0' <= self.name[ksize_edidx] <= '9'):
            ksize_edidx += 1
        return self.name[ksize_stidx:ksize_edidx]

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'REDS')
        # if self.name.lower().find('test') >= 0:
        #     self.dir_hr = os.path.join(self.apath, 'val/val_sharp')
        #     self.dir_lr = os.path.join(self.apath, 'test/test_blur_jpeg')
        if self.name.lower().find('val') >= 0:
            self.dir_hr = os.path.join(self.apath, 'val/val_sharp')
            self.dir_lr = os.path.join(self.apath, 'val/val_blur_jpeg')
        else:
            self.dir_hr = os.path.join(self.apath, 'train/train_sharp')
            self.dir_lr = os.path.join(self.apath, 'train/train_blur_jpeg')
            if self.name.lower().find('ksize') >= 0: # motion blur augmentation
                self.dir_lr += '_ksize{}'.format(self._return_ksize())

            if self.flag_ae_loss:
                self.dir_hr_ae = os.path.join(self.apath, 'train/train_sharpQ25')
            else:
                self.dir_hr_ae = None

        self.ext = ('.png', '.jpg', '.jpg') # exts for HR, LR, HR_AE(this should be changed .jpg)

        if self.args.degradation:
            self.dir_hr, self.dir_lr = self.dir_lr, self.dir_hr
            self.ext = ('.jpg', '.png', '.jpg')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            # Make a folder for each image sequence
            folder_path = '/'.join(f.split('/')[:-1])
            os.makedirs(folder_path, exist_ok=True)
            with open(f, 'wb') as _f:
                pickle.dump(cv2.imread(img, cv2.IMREAD_COLOR)[...,::-1], _f)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        f_hr_ae = self.images_hr_ae[idx] if self.flag_ae_loss else None

        filename, _ = os.path.splitext('/'.join(f_hr.split('/')[-2:]))
        if self.args.ext == 'img' or self.benchmark:
            hr = cv2.imread(f_hr, cv2.IMREAD_COLOR)[...,::-1]
            lr = cv2.imread(f_lr, cv2.IMREAD_COLOR)[...,::-1]
            hr_ae = cv2.imread(f_hr_ae, cv2.IMREAD_COLOR)[...,::-1] if f_hr_ae else None

        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
            if f_hr_ae:
                with open(f_hr_ae, 'rb') as _f:
                    hr_ae = pickle.load(_f)
            else:
                hr_ae = None
        
        if hr_ae is not None:
            hr = np.concatenate((hr, hr_ae), axis=-1)
                

        return lr, hr, filename
