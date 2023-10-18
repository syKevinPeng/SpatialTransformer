import argparse
from sympy import N
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
from network.FlowNetC import FlowNetC

# from network.pwcnet import PWCDCNet
from network.external_packages.torch_receptive_field.receptive_field import (
    receptive_field,
)
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
from torchvision.transforms.functional import InterpolationMode
from icecream import ic 

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
parser.add_argument(
    "--num_img_to_train",
    type=int,
    default=False,
    help="set number of images to train with. This number should be smaller than the total number of images",
)
parser.add_argument("--eval", action="store_true", help="do evaluation")
parser.add_argument("--train", action="store_true", help="do training")
parser.add_argument("--test", action="store_true", help="do testing")
parser.add_argument("--exp_weight", default=0.99)
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    f"--dataset_path", type=str, default="/mnt/e/Downloads/ChairsSDHom/data"
)
parser.add_argument("--network", type=str, default="FlowNetSD")
parser.add_argument("--loss", type=str, default="multi_scale")
parser.add_argument("--to_gray", action="store_true")
parser.add_argument("--name", type=str, default="flownet-exp3")
parser.add_argument("--notes", type=str, default="Whole dataset, rgb images")
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="offline",
    choices=["online", "offline", "disabled"],
)
parser.add_argument(
    "--test_dataset_path",
    type=str,
    default="/home/siyuan/research/SpatialTransformer/data/ouchi_0-255_FLOW",
)

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset(opt):
    train_dset = ChairsSDHom(
        is_cropped=0,
        root=opt.dataset_path,
        dstype="train",
        to_gray=opt.to_gray,
        num_imgs=opt.num_img_to_train,
    )
    val_dest = ChairsSDHom(
        is_cropped=0,
        root=opt.dataset_path,
        dstype="test",
        to_gray=opt.to_gray,
        num_imgs=opt.num_img_to_train,
    )
    test_img_dset = Train_Dataset(
        dir=opt.test_dataset_path, num_imgs=opt.num_img_to_train, to_gray=opt.to_gray
    )
    if len(test_img_dset) == 0:
        raise Exception("no ouchi data found")
    train_DLoader = DataLoader(
        train_dset, batch_size=opt.bat_size, shuffle=True, num_workers=0, pin_memory=0
    )
    val_Dloader = DataLoader(
        val_dest, batch_size=16, shuffle=False, num_workers=0, pin_memory=0
    )
    test_img_DLoader = DataLoader(
        test_img_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=0
    )
    print(f"train set size: {len(train_dset)}")
    return train_DLoader, val_Dloader, test_img_DLoader


def load_network(opt):
    if opt.network == "FlowNetSD":
        if opt.to_gray:
            net = FlowNetSD(
                num_input_chan=2
            ).cuda()  # if gray, the input size would be (batch_size, 2, 256, 256)
            # receptive_field(net, (2, 256, 256))
        else:
            net = (
                FlowNetSD().cuda()
            )  # if rgb, the input size would be (batch_size, 6, 256, 256)
    elif opt.network == "cnn":
        net = CNN(depth=10).cuda()
    elif opt.network == "pwc":
        # md is maximum displacement for correlation
        net = PWCDCNet().cuda()
    elif opt.network == "FlowNetC":
        if opt.to_gray:
            net = FlowNetC(num_input_chan=2).cuda()
            # receptive_field(net, (2, 256, 256))
        else:
            net = FlowNetC(num_input_chan=6).cuda()

    else:
        raise Exception("network not supported")
    print(f"Using {opt.network}")
    return net


def load_loss(opt):
    if opt.loss == "multiscale":
        return MultiScale()
    elif opt.loss == "unsup_loss":
        return UnsupLoss()
    else:
        raise Exception(
            f"loss {opt.loss} not supported. It has to be one of multi_scale, unsup_loss"
        )


def train(opt, net, train_DLoader):
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LR)
    scheduler = MultiStepLR(
        optimizer, milestones=[700, 900, 1300], gamma=0.5
    )  # learning rates

    start_epoch = 0
    if opt.resume or wandb.run.resumed:
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
                    bat_pred_flow = torchvision.transforms.Resize(
                        (256, 256), interpolation=InterpolationMode.BILINEAR
                    )(bat_pred_flow[0])
                    # print(f'bat_pred_flow shape: {bat_pred_flow.shape}')
                    [loss, epe_loss] = load_loss(opt)(
                        bat_img1, bat_img2, bat_pred_flow, bat_gt_flow
                    )
                    loss.backward()
                    optimizer.step()

                if opt.network == "FlowNetC":
                    bat_img1 = bat["im1"].cuda()
                    bat_img2 = bat["im2"].cuda()
                    input_imgs = torch.cat((bat_img1, bat_img2), dim=1)
                    # input_imgs size = (batch_size, 6, 256, 256)

                    bat_gt_flow = bat["flow"].cuda()

                    # bat_pred_flow = net(torch.cat((bat_im1, bat_im2), dim=1))
                    bat_pred_flow = net(input_imgs)
                    # loss return a list [loss value, epe value]
                    [loss, epe_loss] = load_loss(opt)(
                        bat_img1, bat_img2, bat_pred_flow, bat_gt_flow
                    )
                    loss.backward()
                    optimizer.step()

                if opt.network == "FlowNetSD":
                    # (3, 2, 256, 256)=> we need to have input shape (channel, num_img, height, width)
                    bat_img1 = bat["im1"].cuda()
                    bat_img2 = bat["im2"].cuda()
                    input_imgs = torch.cat((bat_img1, bat_img2), dim=1)
                    # input_imgs size = (batch_size, 6, 256, 256)

                    bat_gt_flow = bat["flow"].cuda()

                    # bat_pred_flow = net(torch.cat((bat_im1, bat_im2), dim=1))
                    bat_pred_flow = net(input_imgs)
                    # loss return a list [loss value, epe value]
                    [loss, epe_loss] = load_loss(opt)(
                        bat_img1, bat_img2, bat_pred_flow, bat_gt_flow
                    )
                    loss.backward()
                    optimizer.step()
                wandb.log(
                    {
                        "train/loss": loss.cpu().detach().numpy(),
                        "train/epe": epe_loss.cpu().detach().numpy(),
                        "epoch": start_epoch,
                    }
                )

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
                wandb.save(
                    os.path.join(opt.save_path, "net_epoch_%s.pth" % start_epoch),
                    base_path=opt.save_path,
                )


def validation(opt, net, val_DLoader, viz_flag=False):
    
    net.eval()
    # load pytorch model
    if opt.load_path:
        ckp = torch.load(opt.load_path)
        net.load_state_dict(ckp["net"])
        print(f"load from {opt.load_path}")
    else:
        raise Exception("load path not specified")
    sum_epe = []

    # calculate average epe for the whole dataset
    for bat in val_DLoader:
        with torch.no_grad():
            bat_im1, bat_im2, bat_gt_flow = (
                    bat["im1"].cuda(),
                    bat["im2"].cuda(),
                    bat["flow"].cuda(),
                )
            pred_flow = net(torch.cat((bat_im1, bat_im2), dim=1))
            pred_flow = torchvision.transforms.Resize(
                    (256, 256), interpolation=InterpolationMode.BILINEAR
                )(pred_flow[0])
            # calculate average epe for the batch
            epe = torch.sqrt(torch.sum((pred_flow - bat_gt_flow) ** 2, dim=1))
            epe = torch.mean(epe, dim=(1, 2))
            sum_epe.append(epe.cpu().detach().numpy())
    sum_epe = np.concatenate(sum_epe)
    avg_epe = np.mean(sum_epe)
    wandb.log({"eval/avg_epe": avg_epe})
    print(f"avg_epe: {avg_epe}")

    if viz_flag:
        out_path = Path(opt.save_path) / "viz_result"
        out_path.mkdir(parents=True, exist_ok=True)
    # visulize the flow for the first image
        for bat in val_DLoader:
            with torch.no_grad():
                bat_im1, bat_im2, bat_gt_flow = (
                    bat["im1"].cuda(),
                    bat["im2"].cuda(),
                    bat["flow"].cuda(),
                )
                pred_flow = net(torch.cat((bat_im1, bat_im2), dim=1))
                if opt.network == "pwc" or opt.network == "FlowNetC":
                    pred_flow = torchvision.transforms.Resize(
                            (256, 256), interpolation=InterpolationMode.BILINEAR
                        )(pred_flow[0])
                # remove redundent first channel and swap channels to w, h, c
                count = 0
                for data in zip(bat_im1, bat_im2, bat_gt_flow, pred_flow):
                    im1, im2, gt_flow, pred_flow = data
                    my_pred_flow = pred_flow.cpu().permute(1, 2, 0)
                    img1 = im1.cpu().permute(1, 2, 0)
                    gt_flow = gt_flow.cpu().permute(1, 2, 0)

                    h, w = my_pred_flow.shape[:2]
                    # visulize the flow
                    x, y = np.meshgrid(np.arange(w), np.arange(h))
                    # Get the x and y components of the flow vectors
                    pred_v = my_pred_flow[:, :, 0]
                    pred_u = my_pred_flow[:, :, 1]

                    gt_v = gt_flow[:, :, 0]
                    gt_u = gt_flow[:, :, 1]

                    # Set the arrow spacing and scale
                    spacing = 10
                    scale = None
                    # Create a new image with the original image and flow arrows
                    fig, ax = plt.subplots()
                    ax.imshow(img1)
                    ax.quiver(
                        x[::spacing, ::spacing],
                        y[::spacing, ::spacing],
                        -pred_u[::spacing, ::spacing],
                        pred_v[::spacing, ::spacing],
                        scale=scale,
                        color="r",
                    )
                    ax.quiver(
                        x[::spacing, ::spacing],
                        y[::spacing, ::spacing],
                        -gt_u[::spacing, ::spacing],
                        gt_v[::spacing, ::spacing],
                        scale=scale,
                        color="b",
                    )
                    plt.savefig(out_path / f"viz_{count}.png", dpi=300)

                    # create an endpoint error map (EPE map)
                    # create a new image
                    plt.figure()
                    epe = torch.sqrt(torch.sum((my_pred_flow - gt_flow) ** 2, dim=-1))
                    plt.imshow(epe, cmap="hot", interpolation="nearest")
                    plt.colorbar(label="EPE")
                    plt.title("EPE map")
                    plt.savefig(out_path / f"epe_map_{count}.png", dpi=300)
                    wandb.log(
                        {"epe_map": wandb.Image(str(out_path / f"epe_map_{count}.png"))}
                    )

                    count += 1
                break


def prediction(opt, net, val_DLoader):
    net.eval()
    # load pytorch model
    if opt.load_path:
        ckp = torch.load(opt.load_path)
        if opt.network == "pwc":
            net.load_state_dict(ckp)
        else:
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

            out_path = Path(opt.save_path) / "viz_result"
            out_path.mkdir(parents=True, exist_ok=True)
            pred_flow = net(torch.cat((im1, im2), dim=1))
            # remove redundent first channel and swap channels to w, h, c
            if opt.network == "pwc" or opt.network == "FlowNetC":
                pred_flow = torchvision.transforms.Resize(
                    (256, 256), interpolation=InterpolationMode.BILINEAR
                )(pred_flow[0])
            pred_flow = pred_flow.cpu().squeeze(0).permute(1, 2, 0)
            img1 = im1.cpu().squeeze(0).permute(1, 2, 0)
            gt_flow = gt_flow.cpu().squeeze(0).permute(1, 2, 0)

            if opt.network == "pwc":
                # sacle the flow
                pred_flow = pred_flow * 20
                pred_flow = pred_flow.permute(2, 0, 1)
                print(f"pred_flow shape: {pred_flow.shape}")
                pred_flow = torchvision.transforms.Resize((256, 256))(pred_flow)
                pred_flow = pred_flow.permute(1, 2, 0)

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
            if img1.shape[2] == 1:
                # convert one channel image to three channel
                img1 = img1.repeat(1, 1, 3)
            ax.imshow(img1)
            ax.quiver(
                x[::spacing, ::spacing],
                y[::spacing, ::spacing],
                -pred_u[::spacing, ::spacing],
                pred_v[::spacing, ::spacing],
                scale=scale,
                color="r",
            )
            ax.quiver(
                x[::spacing, ::spacing],
                y[::spacing, ::spacing],
                -gt_u[::spacing, ::spacing],
                gt_v[::spacing, ::spacing],
                scale=scale,
                color="b",
            )
            plt.savefig(out_path / f"viz_{n_count}.png", dpi=300)
            output_imgs.append(out_path / f"viz_{n_count}.png")
            wandb.log({"viz": [wandb.Image(str(img)) for img in output_imgs]})

            pred_flow = pred_flow.numpy()
            gt_flow = gt_flow.numpy()
            # create an endpoint error map (EPE map)
            # create a new image
            plt.figure()
            epe = np.sum((pred_flow - gt_flow) ** 2, axis=-1)
            plt.imshow(epe, cmap="hot", interpolation="nearest")
            plt.colorbar(label="EPE")
            plt.title("EPE map")
            plt.savefig(out_path / f"epe_map_{n_count}.png", dpi=300)
            wandb.log(
                {"epe_map": wandb.Image(str(out_path / f"epe_map_{n_count}.png"))}
            )

            # calculate epe for image:
            imgepe = np.sqrt(epe).flatten()
            avg_epe = np.mean(imgepe)
            wandb.log({"test/avg_epe": avg_epe})
            print(f"avg_epe: {avg_epe}")


1

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
        tags=train_tag,
        # wandb mode
        mode=opt.wandb_mode,
    )

    # also upload key files to wandb
    artifact = wandb.Artifact(
        name="src", type="code", description="contains all source code"
    )
    artifact.add_file("loss.py")
    artifact.add_file("main.py")
    artifact.add_file("network/FlowNetSD.py")
    artifact.add_file("network/pwcnet.py")
    artifact.add_file("train.sh")
    run.log_artifact(artifact)

    setup_seed(0)

    # get dataloader
    train_DLoader, val_Dloader, test_img_DLoader = load_dataset(opt)
    # get network
    net = load_network(opt)

    if opt.train:
        print(f"Begin training")
        train(opt, net, train_DLoader)
        print(f"Finish training")
    if opt.eval:
        print(f"Begin evaluation")
        validation(opt, net, val_Dloader, viz_flag=True)
        print(f"Finish evaluation")
    if opt.test:
        print(f"Begin testing/visulization")
        # prediction on ouchi data
        prediction(opt, net, test_img_DLoader)

        print(f"Finish testing/visulization")
