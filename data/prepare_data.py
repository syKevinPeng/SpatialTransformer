import os
from glob import glob
import numpy as np
import torch
from matplotlib.image import imread

import sys
sys.path.append('../')
from utils.imtools import rgb2gray, imshow
from scipy.io import savemat
from SpatialTransformer import SpatialTransformer
from pathlib import Path
from torchvision import transforms
from matplotlib import pyplot as plt
import imageio, cv2

FILE_NAME = 'texture7'
INPUT_DATA = '/home/siyuan/research/SpatialTransformer/data/texture/texture7.png'
OUTPUT_SIZE = (256, 256)
MOVEMENT = 3
DEVICE = 'cuda:0'



data = sorted(glob(INPUT_DATA))
data_num = 1
save_data_dir = './'+ FILE_NAME +'_FLOW/'
os.makedirs(save_data_dir,exist_ok=True)

def produce_uniform_flow(im1, magnitude = 5):
    # theta = np.random.uniform(low=0,high=360/180 * np.pi)
    theta = -45/180 * np.pi
    # set theta so that x =1 and y = 0
    # theta = 0/180 * np.pi
    # theta = 0
    x = np.cos(theta) * magnitude
    y = np.sin(theta) * magnitude
    im1 = torch.from_numpy(im1)
    if len(im1.shape) == 2:
        img_size = im1.shape[:2]
        im1 = im1.unsqueeze(0).unsqueeze(0)
    elif len(im1.shape) == 3:
        img_size = im1.shape[:2]
        im1 = im1.permute(2,0,1)
        im1 = im1.unsqueeze(0)
    else :
        raise ValueError('The image dimension is wrong!')

    flow = torch.ones((1,2,*img_size))
    flow[0, 0, :, :] = flow[0, 0, :, :] * x
    flow[0, 1, :, :] = flow[0, 1, :, :] * y
    flow = flow.cpu().numpy()


    im2 = torch.zeros_like(im1)
    for i in range(im1.shape[2]):
        for j in range(im1.shape[3]):
            ii = i + flow[0, 0, i, j]
            jj = j + flow[0, 1, i, j]
            if 0 <= ii < im1.shape[2] and 0 <= jj < im1.shape[3]:
                im2[:,:,i,j] = im1[:,:,int(ii),int(jj)]
            else:
                im2[:,:,i,j] = im1[:,:,i,j]
    
    im1, im2, flow = im1.squeeze(0), im2.squeeze(0), flow.squeeze(0)

    # do a center crop of 256x256 using torch transform
    img_transforms = transforms.Compose([
        transforms.CenterCrop(256)])
    
    flow_transforms = transforms.Compose([
        transforms.CenterCrop(256)])    
    
    # convert flow to tensor
    flow = torch.from_numpy(flow)
    flow = flow_transforms(flow)

    im1 = img_transforms(im1)
    im2 = img_transforms(im2)
    # im1 = np.array(im1, dtype=np.float32)
    # im2 = np.array(im2, dtype=np.float32)
    im1 = im1.squeeze(0)
    im2 = im2.squeeze(0)
    # manually crop the flow
    print(f'img1 shape: {im1.shape}, img2 shape: {im2.shape}, flow shape: {flow.shape}')

    # convert image to unit8
    im1 = im1.numpy().astype(np.uint8)
    im2 = im2.numpy().astype(np.uint8)
    
    return im1, im2, flow

def produce_zoom_flow(im1, magnitude = 10):
    # resize to 256x256
    im1 = cv2.resize(im1, (256,256), interpolation=cv2.INTER_AREA)
    center = [im1.shape[0] // 2,im1.shape[1] // 2]
    radius = [0,130]

    # im1 = torch.from_numpy(im1.astype(np.float32)).cuda()
    # im1 = im1.unsqueeze(0).unsqueeze(0)
    im1 = torch.from_numpy(im1)
    if len(im1.shape) == 2:
        img_size = im1.shape[:2]
        im1 = im1.unsqueeze(0).unsqueeze(0).type(torch.float32)
    elif len(im1.shape) == 3:
        img_size = im1.shape[:2]
        im1 = im1.permute(2,0,1)
        im1 = im1.unsqueeze(0).type(torch.float32)
    else :
        raise ValueError('The image dimension is wrong!')
    # im1, im2, flow = im1.squeeze(0), im2.squeeze(0), flow.squeeze(0)
    my_transforms = transforms.Compose([
        transforms.CenterCrop(256)])

    spatial_transform = SpatialTransformer(im1.shape[-2:]).cuda()
    flow = torch.zeros((1,2,*im1.shape[-2:])).cuda()
    ref = torch.ones_like(im1)

    for i in range(im1.shape[2]):
        for j in range(im1.shape[3]):
            ii = i - center[0]
            jj = j - center[1]
            if radius[0]**2 <= ii**2 + jj**2 < radius[1]**2:
                r = np.sqrt(ii**2 + jj**2).astype(int) / center[0] * magnitude
                if ii != 0:
                    theta = np.arctan(np.abs(jj) / np.abs(ii))
                elif ii == 0:
                    theta = 90/180 * np.pi

                flow[0, 0, i, j] = -r * np.cos(theta) * np.sign(ii)
                flow[0, 1, i, j] = -r * np.sin(theta) * np.sign(jj)

    im1 = im1.cuda()
    flow = flow.cuda()
    ref = ref.cuda()

    im2_tmp = spatial_transform(im1, flow)
    ref_tmp = spatial_transform(ref, flow)
    im2_tmp[ref_tmp != 1] = im1[0,0,0,0]
    im2 = im2_tmp

    im1 = my_transforms(im1)
    im2 = my_transforms(im2)
    flow = my_transforms(flow)

    im1 = im1.squeeze(0).type(torch.uint8)
    im2 = im2.squeeze(0).type(torch.uint8)
    flow = flow.squeeze(0).type(torch.uint8)

    # swap channel
    im1 = im1.permute(1,2,0)
    im2 = im2.permute(1,2,0)
    flow = flow.permute(1,2,0)

    return im1, im2, flow


for i in range(data_num):
    # if name == 'Ouchi':
    im1 = imageio.imread(os.path.join(data[0]))
    if im1.shape[-1] == 4:
        im1 = im1[:,:,:3]
    if len(im1.shape) == 3:
        im1 = rgb2gray(im1)
    im1, im2, flow = produce_uniform_flow(im1, magnitude = MOVEMENT)
    # im1, im2, flow = produce_zoom_flow(im1, magnitude = MOVEMENT)
    # put image back to cpu
    if type(im1) == torch.Tensor:
        im1 = im1.cpu().numpy()
        im2 = im2.cpu().numpy()
        flow = flow.cpu().numpy()

    print(f'img1 shape: {im1.shape}, img2 shape: {im2.shape}, flow shape: {flow.shape}')
    imageio.imwrite(save_data_dir + "/im1_%d.png"%i, im1)
    imageio.imwrite(save_data_dir + "/im2_%d.png"%i, im2)
    savemat(save_data_dir + "/mat_%d.mat"%i, {'flow':flow})
    # img = vis_flow(flow.squeeze(), save_fig = True, save_dir= "./data/" + name + "_FLOW/flow_%d.png"%i)
