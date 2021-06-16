# -*- coding: utf-8 -*-
# @Author: Donghyeon Lee (donghyeon1223@gmail.com)
# @Date:   2021-01-10 13:36:50
# @Last Modified by:   Donghyeon Lee (donghyeon1223@gmail.com)
# @Last Modified time: 2021-01-10 13:57:13
import os 

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images