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
import torchvision.transforms as transforms
from SpatialTransformer import SpatialTransformer
from utils import frame_utils
from utils.flow_utils import vis_flow
from utils.imtools import rgb2gray, imshow
import cv2


class Train_Dataset(Dataset):
    def __init__(self, dir, num_imgs = 0, to_gray = False, render_size = (256, 256)):
        self.dir = dir
        self.num_imgs = num_imgs
        self.to_gray = to_gray
        self.data_im1 = sorted(glob(os.path.join(dir, 'im1*.png')))
        self.render_size = render_size


    def  __len__(self):
        if self.num_imgs != 0: self.data_im1 = self.data_im1[0:self.num_imgs]
        img_num = len(self.data_im1)
        return img_num

    def __getitem__(self, i):
        img1 = imread(os.path.join(self.dir, 'im1_%d.png'%i))
        img2 = imread(os.path.join(self.dir, 'im2_%d.png' % i))
        flow = loadmat(os.path.join(self.dir, 'mat_%d.mat' % i))['flow'].astype(np.float32)

        # check if image is one channel, if so, change it to three channels
        if self.to_gray:
            if len(img1.shape) == 3:
                img1 = rgb2gray(img1)
                img2 = rgb2gray(img2)
            else:# the gray gray case
                img1 = img1.reshape(1, img1.shape[0], img1.shape[1])
                img2 = img2.reshape(1, img2.shape[0], img2.shape[1])
        else: # RGB case
            if len(img1.shape) == 2:
                img1 = np.stack((img1, img1, img1), axis=0)
            if len(img2.shape) == 2:
                img2 = np.stack((img2, img2, img2), axis=0)
        flow = torch.from_numpy(flow).squeeze()

        dic = {'im1': img1, 'im2': img2, 'flow':flow}
        return dic
    
class Siyuan_Ouchi_Dataset(Dataset):
    def __init__(self, dir, num_imgs = 0) -> None:
        self.dir = dir
        self.num_imgs = num_imgs
        self.data_im1 = sorted(glob(os.path.join(dir, "image_1", '*_10.png')))
        self.data_im2 = sorted(glob(os.path.join(dir, "image_1", '*_11.png')))
        self.data_flow = sorted(glob(os.path.join(dir, "flow_noc", '*.png')))
        if len(self.data_im1) != len(self.data_im2):
            raise Exception("two set of images don't have same length")
        if len(self.data_flow) == 0:
            raise Exception("no flow found")
    
    def  __len__(self):
        if self.num_imgs != 0: 
            self.data_im1 = self.data_im1[0:self.num_imgs]
            self.data_im2 = self.data_im2[0:self.num_imgs]
        img_num = len(self.data_im1)
        return img_num    
    
    def __getitem__(self, i):
        flow = readFlowKITTI(self.data_flow[i])[0]
        img1 = torch.from_numpy(imread(self.data_im1[i])).permute(2, 0, 1)[0,::]
        img2 = torch.from_numpy(imread(self.data_im2[i])).permute(2, 0, 1)[0,::]
        flow = torch.from_numpy(flow).permute(2, 0, 1)
        result_dict = {'im1': img1.unsqueeze(0), 'im2': img2.unsqueeze(0), 'flow': flow}
        return result_dict


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
  def __init__(self, crop_size=(256,256), render_size=(256,256), is_cropped = False, to_gray = 1,
               root = '/path/to/chairssdhom/data', dstype = 'train', num_imgs=0):
    self.is_cropped = is_cropped
    self.crop_size = crop_size
    self.render_size = render_size
    self.to_gray = to_gray
    self.num_imgs = num_imgs

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.pfm') ) )

    if len(image1) != len(image2) or len(image1) == 0:
        raise Exception(f"images1, length: {len(image1)}, and images2 , length: {len(image2)}, are not the same length or empty:")

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

    if self.num_imgs !=0:
        self.image_list = self.image_list[0:self.num_imgs]
        self.flow_list  = self.flow_list[0:self.num_imgs]


  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = flow[::-1,:,:].copy()
    preprocess_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(self.render_size),
        transforms.ToTensor()
    ])
    preprocess_img_gray = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(self.render_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    preprocess_flow = transforms.Compose([
        transforms.CenterCrop(self.render_size)
    ])
    # images = np.array([img1, img2]).transpose(3,0,1,2)
    
    if self.to_gray:
        img1 = preprocess_img_gray(img1)
        img2 = preprocess_img_gray(img2)
    
    flow = torch.tensor(flow.transpose(2,0,1))
    flow = preprocess_flow(flow)
    img1 = preprocess_img(img1)
    img2 = preprocess_img(img2)
    # img shape: (3, 256, 256)
    dic = {'im1': img1.to(torch.float32), 'im2': img2.to(torch.float32), 'flow': flow.to(torch.float16)}
    return dic


  def __len__(self):
    return len(self.image_list)

# Convert the optical flow from uint16 to float32 and normalize to [-1, 1]
def readFlowKITTI(filename):
    flow = cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

