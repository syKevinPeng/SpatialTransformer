from functools import partial

import torch
from skimage.transform import warp
from torch import nn
import torch.nn.functional as F

from SpatialTransformer import SpatialTransformer
from utils.imtools import imshow


# Photometric loss
def pme_loss(src, flow, dst, norm = 'L1'):
    if norm == 'L1':
        loss_func = l1_loss
    else:
        loss_func = mse_loss
    spatial_transform = SpatialTransformer(src.shape[-2:]).cuda()
    src_warp = spatial_transform(src, flow)
    loss = loss_func(dst, src_warp)
    return loss

def gradient_loss(s, smooth_coef = 1e-2, penalty='L2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if (penalty == 'L2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    d *= smooth_coef
    return d / 2.0

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

def total_loss(im1, im2, pred_flow, gt_flow, verbose = False):
    # return bcc_loss(im1,im2,pred_flow, gt_flow)
    if verbose:
        pme = pme_loss(im1, pred_flow, im2)
        l1 = l1_loss(pred_flow, gt_flow)
        total = pme+l1
        return total, {'pme': pme, 'l1': l1, 'total': total}
    else:
        return pme_loss(im1, pred_flow, im2) + l1_loss(pred_flow, gt_flow)

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()


# Photometric Loss + Smoothness Loss
class UnsupLoss(nn.Module):
    def forward(self, im1, im2, output, target):
    #     pmelossvalue = pme_loss(im1, output[0], im2, norm = 'L1')
    #     gdlossvalue = gradient_loss(output[0], smooth_coef = 0.01, penalty='L2')
    #     lossvalue = pmelossvalue + gdlossvalue
    #     return lossvalue
        if type(output) is tuple:
            pmelossvalue = pme_loss(im1, output[0], im2, norm = 'L1')
            gdlossvalue = gradient_loss(output[0], smooth_coef = 0.01, penalty='L2')
            lossvalue = pmelossvalue + gdlossvalue
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            # lossvalue += self.loss(output, target)
            lossvalue = pme_loss(im1, output[0], im2)
            return  [lossvalue, epevalue]

# Reference from original Flownet2 code: https://github.com/NVIDIA/flownet2-pytorch/blob/master/losses.py
class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue
    

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*EPE(output_, target_)
                lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]
    

# def unsup_loss(im1, im2, output, target):
#     # Brightness constancy loss + gradient smooth loss
#     # Warp the second image to the first image frame using the estimated optical flow  
    
#     # take the first flow due to multi-scale
#     # flow = output[0]
#     flow= output
#     grid = torch.stack([flow[:, 0] / ((im1.shape[3] - 1.0) / 2.0) - 1.0, flow[:, 1] / ((im1.shape[2] - 1.0) / 2.0) - 1.0], dim=3)  
#     warped_im2 = F.grid_sample(im2, grid, mode='bilinear', padding_mode='border')  
  
#     # Compute the brightness consistency loss  
#     brightness_loss = F.l1_loss(warped_im2, im1)
#     # Compute the gradient smoothness loss
#     smooth_loss = gradient_loss(flow, smooth_coef = 0.01, penalty='L2')  
#     return brightness_loss + smooth_loss


