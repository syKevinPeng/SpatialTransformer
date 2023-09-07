import argparse
from sympy import Q
import torch
import warnings
import numpy as np
import random
import os
import wandb

import torchvision
from network.FlowNetSD import FlowNetSD
from network.flow import CNN
from network.pwcnet import PWCDCNet
# from network.pwcnet import PWCDCNet
from utils.flow_utils import vis_flow
from utils.imtools import imshow, vfshown
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import PIL
from dataset import Train_Dataset, ChairsSDHom, Siyuan_Ouchi_Dataset
from loss import MultiScale, UnsupLoss
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_model_summary import summary

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
parser.add_argument("--num_img_to_train", type=int, default=False, help="set number of images to train with. This number should be smaller than the total number of images")
parser.add_argument("--eval", action="store_true", help="do evaluation")
parser.add_argument("--train", action="store_true", help="do training")
parser.add_argument('--test', action='store_true', help='do testing')
parser.add_argument("--exp_weight", default=0.99)
parser.add_argument("-w", "--write", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    f"--dataset_path", type=str, default="/mnt/e/Downloads/ChairsSDHom/data"
)
parser.add_argument("--network", type=str, default="flownet")
parser.add_argument("--loss", type=str, default="multi_scale")
parser.add_argument("--to_gray", action="store_true")
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

def load_dataset(opt):
    train_dset = ChairsSDHom(
        is_cropped=0, root=opt.dataset_path, dstype="train", to_gray=opt.to_gray, debug=debug)
    val_dest = ChairsSDHom(
        is_cropped=0, root=opt.dataset_path, dstype="test", to_gray=opt.to_gray, debug=debug
    )
    test_ouchi_dset = Train_Dataset(dir="data/small_ouchi_FLOW", debug=None)
    test_wheel_dset = Train_Dataset(dir="./data/outer_wheel_FLOW", debug = None)
    if len(test_ouchi_dset) == 0:
        raise Exception("no ouchi data found")
    if len(test_wheel_dset) == 0:
        raise Exception("no wheel data found")
    #val_ouchi_dset = Siyuan_Ouchi_Dataset(dir="/home/siyuan/research/dataset/ouchi_dataset", debug=None)
    train_DLoader = DataLoader(
        train_dset, batch_size=opt.bat_size, shuffle=True, num_workers=0, pin_memory=0
    )
    val_Dloader = DataLoader(
        val_dest, batch_size=1, shuffle=False, num_workers=0, pin_memory=0
    )
    test_ouchi_DLoader = DataLoader(
        test_ouchi_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=0
    )
    test_wheel_DLoader = DataLoader(
        dataset=test_wheel_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=0
    )
    print(f"train set size: {len(train_dset)}")
    return train_DLoader, val_Dloader, test_ouchi_DLoader, test_wheel_DLoader

def load_network(opt):
    if opt.network == "flownet":
        if opt.to_gray:
            net = FlowNetSD(num_input_chan=2).cuda() # if gray, the input size would be (batch_size, 2, 256, 256)
        else:
            net = FlowNetSD().cuda() # if rgb, the input size would be (batch_size, 6, 256, 256)
    elif opt.network == "cnn":
        net = CNN(depth=10).cuda()
    elif opt.network == "pwc":
        # md is maximum displacement for correlation
        net = PWCDCNet().cuda()
        print(summary(net,torch.zeros((1, 6, 256, 256)).cuda(), show_input=True))
        exit()
    else:
        raise Exception("network not supported")
    print(f'Using {opt.network}')
    return net

def load_loss(opt):
    if opt.loss == "multi_scale":
        return  MultiScale(startScale=1, numScales=7, norm="L2")
    elif opt.loss == "unsup_loss":
        return UnsupLoss()
    else:
        raise Exception(f"loss {opt.loss} not supported. It has to be one of multi_scale, unsup_loss")
    

def train(opt, net, train_DLoader):
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LR)
    scheduler = MultiStepLR(optimizer, milestones=[700, 900, 1300], gamma=0.5)  # learning rates

    start_epoch = 0
    if opt.resume and wandb.run.resumed:
        resume_dir = opt.load_path
        ckp = torch.load(resume_dir)
        net.load_state_dict(ckp["net"])
        optimizer.load_state_dict(ckp["optimizer"])
        loss = ckp["loss"]
        start_epoch = ckp["epoch"]
        print(f"load from {resume_dir} at epoch {start_epoch}")

    print(f"Begin training")
    with tqdm(total=opt.epoch - start_epoch, ncols=100, position=0, leave=True) as t:
        for start_epoch in range(start_epoch, opt.epoch):
            scheduler.step(start_epoch)
            # One epoch training
            bat_num = len(train_DLoader)
            for n_count, bat in enumerate(train_DLoader):
                net.train()
                optimizer.zero_grad()
                # swtich from rgb to bgr only for pwcnet
                if opt.network == "pwc":
                    bat_img1 = bat["im1"].cuda()
                    bat_img2 = bat["im2"].cuda()
                    input_imgs = torch.cat((bat_img1, bat_img2), dim=1)
                    bat_gt_flow = bat["flow"].cuda()
                    bat_pred_flow = net(input_imgs)
                    bat_pred_flow = (F.interpolate(bat_pred_flow, size=(256, 256), mode='bilinear', align_corners=False))
                    bat_pred_flow = bat_pred_flow * 20
                    # print(f'bat_pred_flow shape: {bat_pred_flow.shape}')
                    loss = load_loss(opt)(bat_img1, bat_img2, bat_pred_flow, bat_gt_flow)
                    loss.backward()
                    optimizer.step()

                    
                if opt.network == "flownet":
                    #(3, 2, 256, 256)=> we need to have input shape (channel, num_img, height, width)
                    bat_img1 = bat["im1"].cuda()
                    bat_img2 = bat["im2"].cuda()
                    input_imgs = torch.cat((bat_img1, bat_img2), dim=1)
                    # input_imgs size = (batch_size, 6, 256, 256)
                
                    bat_gt_flow = bat["flow"].cuda()

                    # bat_pred_flow = net(torch.cat((bat_im1, bat_im2), dim=1))
                    bat_pred_flow = net(input_imgs)
                    loss = load_loss(opt)(bat_img1, bat_img2, bat_pred_flow, bat_gt_flow)
                    loss.backward()
                    optimizer.step()

                if opt.write:
                    wandb.log({"train/loss": loss.cpu().detach().numpy()[0], 
                               "train/epe": loss.cpu().detach().numpy()[1], 
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

def validation(opt, net, val_DLoader):
    net.eval()
    # load pytorch model
    if opt.load_path:
        ckp = torch.load(opt.load_path)
        net.load_state_dict(ckp["net"])
        print(f"load from {opt.load_path}")
    else:
        raise Exception("load path not specified")
    sum_epe = []
    for epoch_num, bat in enumerate(tqdm(val_DLoader)):
        with torch.no_grad():
            bat_im1, bat_im2, bat_gt_flow = (
                    bat["im1"].cuda(),
                    bat["im2"].cuda(),
                    bat["flow"].cuda(),
                )

            bat_pred_flow = net(torch.cat((bat_im1, bat_im2), dim=1))
            bat_gt_flow = bat_gt_flow.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            bat_pred_flow = bat_pred_flow.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            squared_distance = np.sum((bat_pred_flow - bat_gt_flow) ** 2, axis=2)
            epe = np.sqrt(squared_distance).flatten()
            sum_epe += epe.tolist()
    avg_epe = np.mean(sum_epe)
    if opt.write:
        wandb.log({"val/avg_epe": avg_epe})



def prediction(opt, net, val_DLoader):
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

            img_transform = torchvision.transforms.Compose(
            [torchvision.transforms.CenterCrop((256, 256))]
            )
            im1 = img_transform(im1)
            im2 = img_transform(im2)
            gt_flow = img_transform(gt_flow)

            out_path = Path(opt.save_path) / "viz_result"
            out_path.mkdir(parents=True, exist_ok=True)
            pred_flow = net(torch.cat((im1, im2), dim=1))
            # remove redundent first channel and swap channels to w, h, c
            pred_flow = pred_flow.cpu().squeeze(0).permute(1, 2, 0)
            img1 = im1.cpu().squeeze(0).permute(1, 2, 0)
            gt_flow = gt_flow.cpu().squeeze(0).permute(1, 2, 0)
            if pred_flow.shape[:2] != img1.shape[:2]:
                raise ValueError(
                    f"pred_flow {pred_flow.shape[:2]} and image {img1.shape[:2]} should"
                    " have same shape"
                )
            h, w = pred_flow.shape[:2]
            # visulize the flow
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            # Get the x and y components of the flow vectors
            pred_v = pred_flow[:, :, 0] 
            pred_u = pred_flow[:, :, 1]

            gt_v = gt_flow[:, :, 0]
            gt_u = gt_flow[:, :, 1]

            # Set the arrow spacing and scale
            spacing = 10
            scale = None
            # Create a new image with the original image and flow arrows
            fig, ax = plt.subplots()
            ax.imshow(img1)
            ax.quiver(x[::spacing, ::spacing], y[::spacing, ::spacing], -pred_u[::spacing, ::spacing], pred_v[::spacing, ::spacing], scale=scale, color='r')
            ax.quiver(x[::spacing, ::spacing], y[::spacing, ::spacing], -gt_u[::spacing, ::spacing], gt_v[::spacing, ::spacing], scale=scale, color='b')
            plt.savefig(out_path/f"viz_{n_count}.png", dpi=300)
            output_imgs.append(out_path/f"viz_{n_count}.png")
            if opt.write: wandb.log({"viz": [wandb.Image(str(img)) for img in output_imgs]})


if __name__ == "__main__":
    opt = parser.parse_args()
    torch.cuda.device(opt.gpu_idx)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor

    os.makedirs(opt.save_path, exist_ok=True)

    train_tag = []
    if opt.train:
        train_tag.append("train")
    if opt.eval:
        train_tag.append("eval")
    if opt.test:
        train_tag.append("test")
    
    if opt.write:
        # setup wandb run
        run = wandb.init(
        # Set the project where this run will be logged
        project="Optical Illusion",
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

    # also upload key files to wandb
    if opt.write:
        artifact = wandb.Artifact(name= "src", type="code", description="contains all source code")
        artifact.add_file("loss.py")
        artifact.add_file("main.py")
        artifact.add_file("network/FlowNetSD.py")
        artifact.add_file("network/pwcnet.py")
        artifact.add_file("train.sh")
        run.log_artifact(artifact)

    setup_seed(0)

    if opt.debug:
        debug = 20
    else:
        debug = None

    # get dataloader
    train_DLoader, val_Dloader, test_ouchi_DLoader, test_wheel_DLoader = load_dataset(opt)
    # get network
    net = load_network(opt)

    if opt.train:
        print(f"Begin training")
        train(opt, net, train_DLoader)
        print(f"Finish training")
    if opt.eval:
        print(f"Begin evaluation")
        validation(opt, net, val_Dloader)
        print(f"Finish evaluation")
    if opt.test:
        print(f"Begin testing/visulization")
        # prediction on ouchi data
        prediction(opt,net, test_ouchi_DLoader)
        # prediction on wheel data
        # prediction(opt,net, test_wheel_DLoader)
        print(f"Finish testing/visulization")

    

    


