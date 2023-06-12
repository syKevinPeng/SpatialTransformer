import os
from glob import glob
import numpy as np
import torch
from matplotlib.image import imread

from utils.flow_utils import save_flow, load_flow, vis_flow
from utils.imtools import imshow, rgb2gray, vfshown
from scipy.io import savemat
from skimage.transform import warp
from skimage.transform import EuclideanTransform
from SpatialTransformer import SpatialTransformer


name = 'MovSin_v2'
data_dir = './data/' + 'MovSin' + '/*.png'
data = sorted(glob(data_dir))
data_num = len(data)
save_data_dir = './data/'+ name +'_FLOW/'
os.makedirs(save_data_dir,exist_ok=True)

def produce_uniform_flow(im1):
    theta = np.random.uniform(low=0,high=360/180 * np.pi)
    magnitude = np.random.uniform(low=1, high=3)
    x = np.cos(theta) * magnitude
    y = np.sin(theta) * magnitude

    im1 = torch.from_numpy(im1.astype(np.float32))
    im1 = im1.unsqueeze(0).unsqueeze(0)
    spatial_transform = SpatialTransformer(im1.shape[-2:])
    flow = torch.ones((1,2,*im1.shape[-2:]))
    flow[0, 0, :, :] = flow[0, 0, :, :] * x
    flow[0, 1, :, :] = flow[0, 1, :, :] * y
    im2_tmp = spatial_transform(im1, flow)
    flow = flow.cpu().numpy()

    ##correction
    im2 = im2_tmp
    im2 = torch.ones_like(im2_tmp) * im1[0,0,0,0]
    im2[:,:,1:-1,:-3] = im2_tmp[:,:,1:-1,:-3]

    # y_coords, x_coords = np.meshgrid(np.arange(nr), np.arange(nc),indexing='ij')
    # im2_by_flow = warp(im1, np.array([y_coords - flow[1, :, :], x_coords - flow[0, :, :]]))
    # This is how the grid changes. It should be minus the flow.

    return im2, flow

def produce_zoom_flow(im1):
    center = [im1.shape[0] // 2,im1.shape[1] // 2]
    radius = [0,130]
    magnitude = 10

    im1 = torch.from_numpy(im1.astype(np.float32)).cuda()
    im1 = im1.unsqueeze(0).unsqueeze(0)
    spatial_transform = SpatialTransformer(im1.shape[-2:]).cuda()
    flow = torch.zeros((1,2,*im1.shape[-2:])).cuda()
    ref = torch.ones_like(im1)

    for i in range(im1.shape[2]):
        for j in range(im1.shape[3]):
            ii = i - center[0]
            jj = j - center[1]
            if radius[0]**2 <= ii**2 + jj**2 < radius[1]**2:
                r = np.sqrt(ii**2 + jj**2).astype(np.int) / center[0] * magnitude
                if ii != 0:
                    theta = np.arctan(np.abs(jj) / np.abs(ii))
                elif ii == 0:
                    theta = 90/180 * np.pi

                flow[0, 0, i, j] = -r * np.cos(theta) * np.sign(ii)
                flow[0, 1, i, j] = -r * np.sin(theta) * np.sign(jj)

    im2_tmp = spatial_transform(im1, flow)
    ref_tmp = spatial_transform(ref, flow)
    im2_tmp[ref_tmp != 1] = im1[0,0,0,0]
    im2 = im2_tmp

    flow = flow.cpu().numpy()
    return im2, flow


for i in range(data_num):
    if name == 'Ouchi':
        im1 = imread(os.path.join(data[0]), 0)
        im1 = rgb2gray(im1)
    else:
        im1 = imread(os.path.join(data[i]),0)

    if np.max(im1) > 2:
        im1 = im1 / 255.0

    im2, flow = produce_uniform_flow(im1)

    imshow(im1, str='im1_%d'%i, dir='./data/'+ name +'_FLOW/')
    imshow(im2, str='im2_%d'%i, dir='./data/'+ name +'_FLOW/')
    savemat("./data/"+ name + "_FLOW/mat_%d.mat"%i, {'flow':flow})
    # img = vis_flow(flow.squeeze(), save_fig = True, save_dir= "./data/" + name + "_FLOW/flow_%d.png"%i)
