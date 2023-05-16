import argparse
from sympy import false
import torch
import warnings
import numpy as np
import random
import os
import imageio

from torch.utils.tensorboard import SummaryWriter
from SpatialTransformer import SpatialTransformer
from network.FlowNetSD import FlowNetSD
from utils.flow_utils import vis_flow
from utils.imtools import imshow, vfshown
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Train_Dataset, ChairsSDHom
from loss import MultiScale, pme_loss

parser = argparse.ArgumentParser()
parser.add_argument('--LR', type=float, default=1e-3, help='number of epochs of training')
parser.add_argument('--bat_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=1000, help='batch size')
parser.add_argument('--data_path', type=str, default="../data/lai/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="./checkpoints/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=500, help='frequency to save results')
parser.add_argument('-C', type=int, default=3, help='frequency to save results')
parser.add_argument('--gpu_idx', type=int, default=1)
parser.add_argument('--debug', type=int, default=False)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--exp_weight', default=0.99)
parser.add_argument('-w', '--write', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument(f'--dataset_path', type=str, default='/mnt/e/Downloads/ChairsSDHom/data')
opt = parser.parse_args()

print(opt)
torch.cuda.device(opt.gpu_idx)
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
        copyfile('loss.py', os.path.join(src_dir, 'loss.py'))
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

net = FlowNetSD().cuda()

# train_dset = Train_Dataset(dir = './data/BSDS_FLOW', debug = debug)
train_dset= ChairsSDHom(is_cropped = False, root = '/mnt/e/Downloads/ChairsSDHom/data', dstype = 'train', debug=1000)
# val_bsds_dset = Train_Dataset(dir = './data/BSDS_VAL_FLOW', debug = 1)
# val_train_dset = Train_Dataset(dir = './data/BSDS_FLOW', debug = 1)
val_ouchi_dset = Train_Dataset(dir = './data/Ouchi_FLOW', debug = None)
# val_movsin_dset = Train_Dataset(dir = './data/MovSin_v2_FLOW', debug = None)
# val_wheel_dset = Train_Dataset(dir = './data/Wheel_FLOW', debug = None)
# val_set8_dset = Train_Dataset(dir = './data/Set8_FLOW', debug = 1)
# val_train_dset= ChairsSDHom(is_cropped = 0, root = '../flownet2_pytorch/data/ChairsSDHom/data', dstype = 'train', replicates = 1, debug=1)


train_DLoader = DataLoader(train_dset, batch_size=opt.bat_size, shuffle=True, num_workers=0, pin_memory=False)
# val_bsds_DLoader = DataLoader(val_bsds_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
val_ouchi_DLoader = DataLoader(val_ouchi_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
# val_train_DLoader = DataLoader(val_train_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
# val_movsin_DLoader = DataLoader(val_movsin_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
# val_wheel_DLoader = DataLoader(val_wheel_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
# val_set8_DLoader = DataLoader(val_set8_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.LR)
scheduler = MultiStepLR(optimizer, milestones=[700, 900], gamma=0.5)  # learning rates

start = 0
if opt.resume:
    resume_dir = './pretrained_model/wheel/net_epoch_850.pth'
    start = 990
    ckp = torch.load(resume_dir, map_location=lambda storage, loc: storage.cuda(opt.gpu_idx))
    net.load_state_dict(ckp['net'])
    optimizer.load_state_dict(ckp['optimizer'])


def val(val_DLoader, name, step):
    net.eval()
    for n_count, bat in enumerate(val_DLoader):
        with torch.no_grad():
            # bat = next(iter(val_DLoader))
            sample_rate = 10
            keep_scale = False
            scale = 5
            im1, im2, gt_flow = bat['im1'].cuda(), bat['im2'].cuda(), bat['flow'].cuda()
            if name == 'ouchi':
                import torch.nn.functional as F
                im1 = F.pad(im1, (0, 0, 11, 12))[:, :, :, 39:-38]
                im2 = F.pad(im2, (0, 0, 11, 12))[:, :, :, 39:-38]
                gt_flow = F.pad(gt_flow, (0, 0, 11, 12))[:, :, :, 39:-38]
                imshow(im1,'im1_%d'%n_count, dir = "./result/ouchi_data")
                imshow(im2, 'im2_%d' % n_count, dir = "./result/ouchi_data")
            pred_flow = net(torch.cat((im1, im2), dim=1))
            if name == 'wheel':
                def compute_gradient(img):
                    import torch.nn.functional as F
                    gradx = img[..., 1:, :] - img[..., :-1, :]
                    grady = img[..., 1:] - img[..., :-1]
                    gradx = F.pad(gradx, (0, 0, 0, 1))
                    grady = F.pad(grady, (0, 1, 0, 0))
                    mag = torch.sqrt(gradx ** 2 + grady ** 2)
                    return mag
                mag = compute_gradient(im1)
                loc = (mag < 0.005).cpu().numpy()

                pred_flow[0,...] = pred_flow[0, ...].cpu() * torch.from_numpy(~loc)
                gt_flow[0, ...] = gt_flow[0, ...].cpu() * torch.from_numpy(~loc)
                sample_rate = 5
                keep_scale = True
                scale = 15


            if name == 'movsin':
                a2 = im2.cpu().numpy().squeeze()
                loc = (a2 >= 0.4)
                pred_flow[0,...] = pred_flow[0, ...].cpu() * torch.from_numpy(~loc)
                gt_flow[0, ...] = gt_flow[0, ...].cpu() * torch.from_numpy(~loc)


            vfshown(pred_flow[:, 0, :, :], pred_flow[:, 1, :, :], sample_rate=sample_rate, save_fig=False,
                    file_name=os.path.join(opt.save_path + 'pre_%d_%s_%d_flow' % (n_count, name, step)),keepscale=keep_scale, scale = scale)
            vfshown(gt_flow[:, 0, :, :], gt_flow[:, 1, :, :], sample_rate=sample_rate, save_fig=False,
                    file_name=os.path.join(opt.save_path + 'gt_%d_%s_flow' % (n_count,name)),keepscale=keep_scale, scale = scale)

            img = vis_flow(gt_flow.cpu().numpy().squeeze())
            imageio.imsave(os.path.join(opt.save_path + 'gt_%d_%s_color.png' % (n_count,name)), img)

            img = vis_flow(pred_flow.cpu().numpy().squeeze())
            imageio.imsave(os.path.join(opt.save_path + 'pre_%d_%s_%d_color.png' % (n_count, name, step)), img)


            # Draw the error plot in term of angles
            if name == 'wheel':
                from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
                struct = generate_binary_structure(2, 1)
                loc = binary_dilation(~loc.squeeze(), struct)
                # loc = binary_dilation(loc, struct)

                a1 = pred_flow.cpu().numpy().squeeze()
                a2 = gt_flow.cpu().numpy().squeeze()
                a1_unit = a1 / (np.sqrt(a1[0,...] ** 2 + a1[1,...] **2))
                a2_unit = a2 / (np.sqrt(a2[0,...] ** 2 + a2[1,...] **2))
                # a2_unit[np.isnan(a2_unit)] = 0

                diff = (a1_unit - a2_unit) * (loc).squeeze()
                # keep_scale = False
                # scale = 5

                vfshown(diff[0, :, :], diff[1, :, :], sample_rate=5, save_fig=True,
                        file_name=os.path.join(opt.save_path + 'pre_%d_%s_%d_diff' % (n_count, name, step)))


            if name == 'train':
                spatial_transform = SpatialTransformer(im1.shape[-2:]).cuda()
                im1_warp = spatial_transform(im1, pred_flow)
                imshow(im1,dir=opt.save_path, str='im1')
                imshow(im2, dir=opt.save_path, str='im2')
                imshow(im1_warp, dir=opt.save_path, str='im1_warp_%d'%step)

            if opt.write:
                try:
                    [loss, epe] = total_loss(im1, im2, pred_flow, gt_flow)
                    writer.add_scalar('Test_%s/loss' % name, loss.cpu().detach().numpy(), iters)
                    writer.add_scalar('Test_%s/epe' % name, epe.cpu().detach().numpy(), iters)
                except:
                    pass

if opt.eval:
    opt.save_path = './tmp/FastView/'
    # val(val_wheel_DLoader, 'wheel', -1)
    val(val_ouchi_DLoader, 'ouchi', -1)
    exit(0)

if opt.resume:
    pass
total_loss = MultiScale(startScale=1,numScales=7,norm= 'L2')

with tqdm(total=opt.epoch - start, ncols=100, position=0, leave=True) as t:
    for epoch in range(start, opt.epoch):
        scheduler.step(epoch)
        # One epoch training
        bat_num = len(train_DLoader)
        for n_count, bat in enumerate(train_DLoader):
            net.train()
            optimizer.zero_grad()

            bat_im1, bat_im2, bat_gt_flow = bat['im1'].cuda(), bat['im2'].cuda(), bat['flow'].cuda()

            bat_pred_flow = net(torch.cat((bat_im1, bat_im2),dim=1))
            loss, list = total_loss(bat_im1, bat_im2, bat_pred_flow, bat_gt_flow)

            loss.backward()
            optimizer.step()

            iters = epoch * bat_num + n_count
            if opt.write:
                # for name in list.keys():
                #     writer.add_scalar('Train/%s'%name, list[name].cpu().detach().numpy(),iters)
                writer.add_scalar('Train/loss', loss.cpu().detach().numpy(), iters)
                writer.flush()


            # Do Validation in several runs.
            if  (n_count == 0) or (opt.debug and epoch % 250 == 0):
                # val(val_bsds_DLoader, 'bsds', epoch)
                val(val_ouchi_DLoader, 'ouchi', epoch)
                # val(val_train_DLoader, 'train', epoch)
                # val(val_movsin_DLoader, 'movsin', epoch)
                # val(val_wheel_DLoader, 'wheel', epoch)

    
        t.set_postfix(loss='%1.3e' % loss.detach().cpu().numpy())
        t.update()
        if epoch % opt.save_frequency == 0 or epoch == opt.epoch - 1:
            state = {'net'       : net.state_dict(),
                     'optimizer' : optimizer.state_dict(),
                     'scheduler' : scheduler.state_dict()}
            torch.save(state, os.path.join(opt.save_path, "net_epoch_%s.pth"%epoch))

