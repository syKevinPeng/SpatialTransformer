from functools import partial

import torch
from skimage.transform import warp
from torch import nn
import torch.nn.functional as F

from SpatialTransformer import SpatialTransformer
from utils.imtools import imshow


def pme_loss(src, flow, dst, norm = 'L2'):
    if norm == 'L1':
        loss_func = l1_loss
    else:
        loss_func = mse_loss
    spatial_transform = SpatialTransformer(src.shape[-2:]).cuda()
    src_warp = spatial_transform(src, flow)
    loss = loss_func(dst, src_warp)
    # imshow(src_warp,'warped')
    # imshow(src, 'src')
    # imshow(dst,'dst')
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

class MultiScale(nn.Module):
    def __init__(self, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)]).cuda()
        self.l_type = norm
        self.div_flow = 1
        assert(len(self.loss_weights) == self.numScales)


    def forward(self, im1, im2, output, target):
        epevalue = 0

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


