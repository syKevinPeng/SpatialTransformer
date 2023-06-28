import argparse
import torch
import warnings
import numpy as np
import random
import os
import wandb

from torch.utils.tensorboard import SummaryWriter
from SpatialTransformer import SpatialTransformer
from network.FlowNetSD import FlowNetSD
from network.flow import CNN
from utils.flow_utils import vis_flow
from utils.imtools import imshow, vfshown
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from dataset import Train_Dataset, ChairsSDHom
from loss import MultiScale, pme_loss, total_loss
import torch.nn.functional as F
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--LR", type=float, default=1e-3, help="number of epochs of training"
)
parser.add_argument("--bat_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=1500, help="epoch size")
parser.add_argument(
    "--data_path", type=str, default="../data/lai/", help="path to blurry image"
)
parser.add_argument(
    "--save_path", type=str, default="./checkpoints/", help="path to save results"
)
parser.add_argument(
    "--save_frequency", type=int, default=1, help="frequency to save results"
)
parser.add_argument("-C", type=int, default=3, help="frequency to save results")
parser.add_argument("--gpu_idx", type=int, default=1)
parser.add_argument("--debug", type=int, default=False)
parser.add_argument("--eval", action="store_true", help="do evaluation")
parser.add_argument("--train", action="store_true", help="do training")
parser.add_argument("--exp_weight", default=0.99)
parser.add_argument("-w", "--write", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    f"--dataset_path", type=str, default="/mnt/e/Downloads/ChairsSDHom/data"
)
parser.add_argument("--network", type=str, default="flownet")
parser.add_argument("--to_gray", type=bool, default=True)
parser.add_argument('--name', type=str, default='flownet-exp3')    
parser.add_argument("--notes", type=str, default="Whole dataset, rgb images")
parser.add_argument("--load_path", type=str, default=None)

warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def save_files(opt):
    if opt.write:
        from shutil import copyfile, copytree

        src_dir = Path(opt.save_path, "src")
        if not src_dir.is_dir():
            os.makedirs(src_dir, exist_ok=True)
            copyfile(
                os.path.basename(__file__),
                os.path.join(src_dir, os.path.basename(__file__)),
            )
            copyfile("loss.py", os.path.join(src_dir, "loss.py"))
            copytree("./network/", os.path.join(src_dir, "network"))
            copytree("./utils/", os.path.join(src_dir, "utils"))

def load_dataset(opt):
    train_dset = ChairsSDHom(
        is_cropped=0, root=opt.dataset_path, dstype="train", to_gray=opt.to_gray, debug=debug)
    val_ouchi_dset = Train_Dataset(dir="./data/Ouchi_FLOW", debug=None)
    train_DLoader = DataLoader(
        train_dset, batch_size=opt.bat_size, shuffle=True, num_workers=0, pin_memory=0
    )
    val_ouchi_DLoader = DataLoader(
        val_ouchi_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=0
    )
    return train_DLoader, val_ouchi_DLoader

def load_network(opt):
    if opt.network == "flownet":
        net = FlowNetSD().cuda()
    elif opt.network == "cnn":
        net = CNN(depth=10).cuda()
    else:
        raise Exception("network not supported")
    print(f'Using {opt.network}')
    return net

def train(opt, net, train_DLoader):
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LR)
    scheduler = MultiStepLR(optimizer, milestones=[700, 900, 1300], gamma=0.5)  # learning rates

    start_epoch = 0
    if wandb.run.resumed:
        resume_dir = opt.load_path
        ckp = torch.load(resume_dir)
        net.load_state_dict(ckp["net"])
        optimizer.load_state_dict(ckp["optimizer"])
        loss = ckp["loss"]
        start_epoch = ckp["epoch"]
        print(f"load from {resume_dir} at epoch {start_epoch}")

    flownet_loss = MultiScale(startScale=1, numScales=7, norm="L2")
    cnn_loss = total_loss

    print(f"Begin training")
    with tqdm(total=opt.epoch - start_epoch, ncols=100, position=0, leave=True) as t:
        for start_epoch in range(start_epoch, opt.epoch):
            scheduler.step(start_epoch)
            # One epoch training
            bat_num = len(train_DLoader)
            for n_count, bat in enumerate(train_DLoader):
                net.train()
                optimizer.zero_grad()

                bat_im1, bat_im2, bat_gt_flow = (
                    bat["im1"].cuda(),
                    bat["im2"].cuda(),
                    bat["flow"].cuda(),
                )

                bat_pred_flow = net(torch.cat((bat_im1, bat_im2), dim=1))
                if opt.network == "flownet":
                    loss, list = flownet_loss(bat_im1, bat_im2, bat_pred_flow, bat_gt_flow)
                elif opt.network == "cnn":
                    loss, list = cnn_loss(bat_im1, bat_im2, bat_pred_flow, bat_gt_flow, verbose=True)
                loss.backward()
                optimizer.step()

                iters = start_epoch * bat_num + n_count
                if opt.write:
                    # for name in list.keys():
                    #     writer.add_scalar('Train/%s'%name, list[name].cpu().detach().numpy(),iters)
                    wandb.log({"train/loss": loss.cpu().detach().numpy(), 
                               "epoch": start_epoch})

            t.set_postfix(loss="%1.3e" % loss.detach().cpu().numpy())
            t.update()
            if start_epoch % opt.save_frequency == 0 or start_epoch == opt.epoch - 1:
                state = {
                    "net": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": start_epoch,
                    "loss": loss,
                }
                torch.save(
                    state, os.path.join(opt.save_path, "net_epoch_%s.pth" % start_epoch)
                )
                # save to wandb
                wandb.save(os.path.join(opt.save_path, "net_epoch_%s.pth" % start_epoch))


def val(opt, net, val_DLoader, name):
    net.eval()
    # load pytorch model
    if opt.load_path:
        ckp = torch.load(opt.load_path)
        net.load_state_dict(ckp["net"])
        print(f"load from {opt.load_path}")
    else:
        raise Exception("load path not specified")
    output_imgs = []
    for n_count, bat in enumerate(val_DLoader):
        with torch.no_grad():
            # bat = next(iter(val_DLoader))
            sample_rate = 10
            keep_scale = False
            scale = 5
            im1, im2, gt_flow = bat["im1"].cuda(), bat["im2"].cuda(), bat["flow"].cuda()
            if name == "ouchi":
                # pad images
                im1 = F.pad(im1, (0, 0, 11, 12))[:, :, :, 39:-38]
                im2 = F.pad(im2, (0, 0, 11, 12))[:, :, :, 39:-38]
                
                out_path = Path(opt.save_path) / "viz_result"
                out_path.mkdir(parents=True, exist_ok=True)
                pred_flow = net(torch.cat((im1, im2), dim=1))
                # remove redundent first channel and swap channels to w, h, c
                pred_flow = pred_flow.cpu().squeeze(0).permute(1, 2, 0)
                # also get the image with padded border
                img1 = im1.cpu().squeeze(0).permute(1, 2, 0)

                # pad flow
                gt_flow = F.pad(gt_flow, (0, 0, 11, 12))[:, :, :, 39:-38]
                # swap axis to w, h, c
                gt_flow = gt_flow.cpu().squeeze(0).permute(1, 2, 0)
                # # resize to output img size 
                # print(pred_flow.shape, gt_flow.shape)
                # exit()
                # gt_flow = F.interpolate(gt_flow, size=pred_flow.shape[:2], mode="nearest").squeeze(0)
                if pred_flow.shape[:2] != img1.shape[:2]:
                    raise ValueError(
                        f"pred_flow {pred_flow.shape[:2]} and image {img1.shape[:2]} should"
                        " have same shape"
                    )
                w, h = pred_flow.shape[:2]
                # visulize the flow
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                # Get the x and y components of the flow vectors
                pred_u = pred_flow[:, :, 0]
                pred_v = pred_flow[:, :, 1]

                gt_u = gt_flow[:, :, 0]
                gt_v = gt_flow[:, :, 1]

                # Set the arrow spacing and scale
                spacing = 10
                scale = None
                # Create a new image with the original image and flow arrows
                fig, ax = plt.subplots()
                ax.imshow(img1)
                ax.quiver(x[::spacing, ::spacing], y[::spacing, ::spacing], -pred_u[::spacing, ::spacing], pred_v[::spacing, ::spacing], scale=scale, color='r')
                ax.quiver(x[::spacing, ::spacing], y[::spacing, ::spacing], -gt_u[::spacing, ::spacing], gt_v[::spacing, ::spacing], scale=scale, color='b')
                plt.savefig(out_path/f"flownet_viz_gray_{n_count}.png", dpi=300)
                output_imgs.append(out_path/f"flownet_viz_gray_{n_count}.png")
            if opt.write: wandb.log({"viz": [wandb.Image(str(img)) for img in output_imgs]})

            if name == "wheel":

                def compute_gradient(img):
                    import torch.nn.functional as F

                    gradx = img[..., 1:, :] - img[..., :-1, :]
                    grady = img[..., 1:] - img[..., :-1]
                    gradx = F.pad(gradx, (0, 0, 0, 1))
                    grady = F.pad(grady, (0, 1, 0, 0))
                    mag = torch.sqrt(gradx**2 + grady**2)
                    return mag

                mag = compute_gradient(im1)
                loc = (mag < 0.005).cpu().numpy()

                pred_flow[0, ...] = pred_flow[0, ...].cpu() * torch.from_numpy(~loc)
                gt_flow[0, ...] = gt_flow[0, ...].cpu() * torch.from_numpy(~loc)
                sample_rate = 5
                keep_scale = True
                scale = 15


if __name__ == "__main__":
    opt = parser.parse_args()

    torch.cuda.device(opt.gpu_idx)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor

    if opt.write:
        os.makedirs(opt.save_path, exist_ok=True)

    train_tag = []
    if opt.train:
        train_tag.append("train")
    if opt.eval:
        train_tag.append("eval")
    
    if opt.write:
        # setup wandb run
        run = wandb.init(
        # Set the project where this run will be logged
        project="SpatialTransformer",
        # name of the experiment
        name=opt.name,
        # notes
        notes=opt.notes,
        # Track hyperparameters and run metadata
        config=opt,
        # flag to resume
        resume=opt.resume,
        # add tags
        tags=train_tag)

    setup_seed(0)
    save_files(opt)

    if opt.debug:
        debug = 20
    else:
        debug = None

    # get dataloader
    train_DLoader, val_ouchi_DLoader = load_dataset(opt)
    # get network
    net = load_network(opt)

    if opt.train:
        print(f"Begin training")
        train(opt, net, train_DLoader)
        print(f"Finish training")
    if opt.eval:
        print(f"Begin evaluation")
        val(opt,net, val_ouchi_DLoader, "ouchi")
        print(f"Finish evaluation")
        exit(0)
    

    


