import argparse
import torch
import warnings
import numpy as np
import random
import os

from skimage.transform import warp
from torch.utils.tensorboard import SummaryWriter

from utils.imtools import imshow, vfshown
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Train_Dataset
from loss import total_loss, EPE
from network.flow import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--LR', type=float, default=1e-3, help='number of epochs of training')
parser.add_argument('--bat_size', type=int, default=16, help='batch size')
parser.add_argument('--epoch', type=int, default=1500, help='batch size')
parser.add_argument('--data_path', type=str, default="../data/lai/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="./tmp/01_4_LongTrain/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=50, help='frequency to save results')
parser.add_argument('-C', type=int, default=3, help='frequency to save results')
parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--debug', type=int, default=False)
parser.add_argument('--exp_weight', default=0.99)
parser.add_argument('--write', action='store_true')
parser.add_argument('--resume', default=False)
opt = parser.parse_args()

print(opt)
torch.cuda.set_device(opt.gpu_idx)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

def save_files(opt):
    if opt.write:
        from shutil import copyfile, copytree
        src_dir = os.path.join(opt.save_path, 'src')
        os.makedirs(src_dir, exist_ok=True)
        copyfile(os.path.basename(__file__), os.path.join(src_dir, os.path.basename(__file__)))
        copytree('./network/', os.path.join(src_dir, 'network'))
        copytree('./utils/', os.path.join(src_dir, 'utils'))
if opt.write:
    os.makedirs(opt.save_path, exist_ok=True)
    writer = SummaryWriter(opt.save_path)

save_files(opt)

if opt.debug:
    debug = 20
else:
    debug = None

net = CNN(depth=10).cuda()

train_dset = Train_Dataset(dir = './data/BSDS_FLOW', debug = debug)
val_bsds_dset = Train_Dataset(dir = './data/BSDS_VAL_FLOW', debug = 1)
val_train_dset = Train_Dataset(dir = './data/BSDS_FLOW', debug = 1)
val_ouchi_dset = Train_Dataset(dir = './data/Ouchi_FLOW', debug = None)

train_DLoader = DataLoader(train_dset, batch_size=opt.bat_size, shuffle=True, num_workers=0, pin_memory=False)
val_bsds_DLoader = DataLoader(val_bsds_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
val_ouchi_DLoader = DataLoader(val_ouchi_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.LR)
scheduler = MultiStepLR(optimizer, milestones=[1000, 1300], gamma=0.5)  # learning rates

start = 0
if opt.resume:
    resume_dir = './tmp/01_1_LR-1e3/net_epoch_27.pth'
    start = 27
    ckp = torch.load(resume_dir, map_location=lambda storage, loc: storage.cuda(opt.gpu_idx))
    net.load_state_dict(ckp['net'])
    optimizer.load_state_dict(ckp['optimizer'])

if opt.resume:
    pass

with tqdm(total=opt.epoch - start, ncols=100, position=0, leave=True) as t:
    for epoch in range(start, opt.epoch):
        scheduler.step(epoch)
        # One epoch training
        print(epoch)
        bat_num = len(train_DLoader)
        for n_count, bat in enumerate(train_DLoader):
            net.train()
            optimizer.zero_grad()

            bat_im1, bat_im2, bat_gt_flow = bat['im1'].cuda(), bat['im2'].cuda(), bat['flow'].cuda()
            bat_pred_flow = net(bat_im1, bat_im2)
            loss, list = total_loss(bat_im1, bat_im2, bat_pred_flow, bat_gt_flow, verbose=True)
            loss.backward()
            optimizer.step()

            iters = epoch * bat_num + n_count
            if opt.write:
                for name in list.keys():
                    writer.add_scalar('Train/%s'%name, list[name].cpu().detach().numpy(),iters)
                writer.flush()


            # Do Validation in several runs.
            if  (n_count == 0) or (opt.debug and n_count % 100 == 0):
                def val(val_DLoader, name, step):
                    net.eval()
                    # for n_count, bat in enumerate(val_DLoader):
                    with torch.no_grad():
                        bat = next(iter(val_DLoader))
                        im1, im2, gt_flow = bat['im1'].cuda().unsqueeze(0), bat['im2'].cuda().unsqueeze(0), bat['flow'].cuda().unsqueeze(0)
                        pred_flow = net(im1, im2)
                        vfshown(pred_flow[:,0,:,:], pred_flow[:,1,:,:], sample_rate=10, save_fig=True,
                                file_name=os.path.join(opt.save_path + 'pre_flow_%s_%d'%(name, step)))
                        vfshown(gt_flow[:, 0, :, :], gt_flow[:, 1, :, :], sample_rate=10, save_fig=True,
                                file_name=os.path.join(opt.save_path + 'gt_flow_%s'%(name)))
                        if opt.write:
                            loss, list = total_loss(im1, im2, pred_flow, gt_flow, verbose=True)
                            epe = EPE(pred_flow, gt_flow)
                            for loss_name in list.keys():
                                writer.add_scalar('Test_%s/%s' % (name,loss_name), list[loss_name].cpu().detach().numpy(), iters)
                            writer.add_scalar('Test_%s/epe'%name, epe.cpu().detach().numpy(), iters)

                val(val_bsds_dset, 'bsds', epoch)
                val(val_ouchi_dset, 'ouchi', epoch)
                val(val_train_dset, 'train', epoch)


        t.set_postfix(loss='%1.3e' % loss.detach().cpu().numpy())
        t.update()

        if epoch % opt.save_frequency == 0 or epoch == opt.epoch - 1:
            state = {'net'       : net.state_dict(),
                     'optimizer' : optimizer.state_dict(),
                     'scheduler' : scheduler.state_dict()}
            torch.save(state, os.path.join(opt.save_path, "net_epoch_%s.pth"%epoch))

