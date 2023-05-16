import os
from glob import glob
from os.path import join

import random

import imageio
from matplotlib.image import imread
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from SpatialTransformer import SpatialTransformer
from utils import frame_utils
from utils.flow_utils import vis_flow
from utils.imtools import rgb2gray, imshow


class Train_Dataset(Dataset):
    def __init__(self, dir, debug = None):
        self.dir = dir
        self.debug = debug

        self.data_im1 = sorted(glob(os.path.join(dir, 'im1*.png')))


    def  __len__(self):
        if self.debug is not None: self.data_im1 = self.data_im1[0:self.debug]
        img_num = len(self.data_im1)
        return img_num

    def __getitem__(self, i):
        im1 = imread(os.path.join(self.dir, 'im1_%d.png'%i))
        im2 = imread(os.path.join(self.dir, 'im2_%d.png' % i))
        flow = loadmat(os.path.join(self.dir, 'mat_%d.mat' % i))['flow'].astype(np.float32)


        im1 = torch.from_numpy(im1).unsqueeze(0)
        im2 = torch.from_numpy(im2).unsqueeze(0)
        flow = torch.from_numpy(flow).squeeze()

        dic = {'im1': im1, 'im2': im2, 'flow':flow}
        return dic

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class ChairsSDHom(Dataset):
  def __init__(self, crop_size=(256,256), render_size=(256,256), is_cropped = 0, to_gray = 1,
               root = '/path/to/chairssdhom/data', dstype = 'train', debug=None):
    self.is_cropped = is_cropped
    self.crop_size = crop_size
    self.render_size = render_size
    self.to_gray = to_gray

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.flo') ) )

    assert (len(image1) == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = image1[i]
        im2 = image2[i]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    self.debug = debug
    if self.debug is not None:
        self.image_list = self.image_list[0:self.debug]
        self.flow_list  = self.flow_list[0:self.debug]


  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = flow[::-1,:,:]
    flow = np.flipud(flow)
    # img = vis_flow(flow)
    # imageio.imsave(os.path.join('tmp' , 'flow.png'), img)
    # imshow(img1, 'im1')
    # imshow(img2, 'im2')


    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)

    if self.to_gray:
        img1 = rgb2gray(images[0])
        img2 = rgb2gray(images[1])

    flow = flow.transpose(2,0,1)


    im1 = torch.from_numpy(img1).unsqueeze(0)
    im2 = torch.from_numpy(img2).unsqueeze(0)
    flow = torch.from_numpy(flow.copy()).squeeze()


    dic = {'im1': im1, 'im2': im2, 'flow': flow}
    return dic


  def __len__(self):
      if self.debug is not None:
          return self.debug
      else:
          return self.size


