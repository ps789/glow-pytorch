from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import sys
from model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=100, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint")
parser.add_argument(
    "--energy_distance", action="store_true", help="use energy distance"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("--path", metavar="PATH", type=str, help="Path to image directory")
parser.add_argument("--n_channels", default = 3, type = int, help = "number of channels")
parser.add_argument("--epoch", type = str, help = "epoch number to load")

def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            #transforms.CenterCrop(image_size),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    #dataset = datasets.ImageFolder(path, transform=transform)
    if path == "mnist":
        dataset = datasets.MNIST(root=path, train=True, transform=transform, download=True)
    
    else:
        dataset = datasets.CIFAR10(root=path, train=True, transform = transform, download = True)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * args.n_channels

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

'''
f_loss: function with as input the (x,y,reuse=False), and as output a list/tuple whose first element is the loss.
'''
def makeScaleMatrix(num_gen, num_orig):
        # first 'N' entries have '1/N', next 'M' entries have '-1/M'
        s1 =  torch.ones(num_gen, 1).to(device)/num_gen
        s2 = -torch.ones(num_orig, 1).to(device)/num_orig
        # 50 is batch size but hardcoded
        return torch.cat([s1, s2], dim=0)

def _mmd_loss1(x, gen_x, sigma = [2, 5, 10, 20, 40, 80]):
        # concatenation of the generated images and images from the dataset
        # first 'N' rows are the generated ones, next 'M' are from the data
        X = torch.cat([gen_x, x], dim=0)
        # dot product between all combinations of rows in 'X'
        XX = torch.matmul(X, torch.transpose(X, 0, 1))
        # dot product of rows with themselves
        X2 = torch.sum(X * X, dim = 1, keepdim=True)
        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
        exponent = XX - 0.5 * X2 - 0.5 * torch.transpose(X2, 0, 1)
        # scaling constants for each of the rows in 'X'
        s = makeScaleMatrix(x.shape[0], x.shape[0])
        # scaling factors of each of the kernel values, corresponding to the
        # exponent values
        S = torch.matmul(s, torch.transpose(s, 0, 1))
        loss = 0
        # for each bandwidth parameter, compute the MMD value and add them all
        for i in range(len(sigma)):
            # kernel values for each combination of the rows in 'X'
            kernel_val = torch.exp(1.0 / sigma[i] * exponent)
            loss += torch.sum(S * kernel_val)
        return loss

def train(args, model, optimizer):
    torch.autograd.set_detect_anomaly(True)
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(args.n_channels, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    losses = []
    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            image = image * 255.

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5
            if not args.energy_distance:
                if i == 0:
                    with torch.no_grad():
                        log_p, logdet, _ = model(
                            image + torch.rand_like(image) / n_bins
                        )

                        continue

                else:
                    log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

                logdet = logdet.mean()

                loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            else:
                z_sample_2 = []
                for z in z_shapes:
                    z_new = torch.randn(args.batch, *z) * args.temp
                    z_sample_2.append(z_new.to(device))
                sample = model.reverse(z_sample_2)
                loss = _mmd_loss1(torch.flatten(image, 1, -1), torch.flatten(sample, 1, -1))
            model.zero_grad()
            if loss >= 10000:
                print("loss large", flush = True)
                print("sample max")
                print(sample.max())
            if torch.isnan(loss):
                print("loss nan", flush = True)
                print("sample max")
                print(sample.max())
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            loss.backward()
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
            if len(losses)>50:
                losses.remove(losses[0])

            if not args.energy_distance:
                pbar.set_description(
                    f"Loss: {sum(losses)/len(losses):.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
                )
            else:
                pbar.set_description(
                    f"Loss: {sum(losses)/len(losses):.5f}; lr: {warmup_lr:.7f}"
                )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"sample/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % 10000 == 0:
                torch.save(
                    model.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        args.n_channels, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = model_single#nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.checkpoint_path != None:
        model.load_state_dict(torch.load(args.checkpoint_path+"/model_"+args.epoch+".pt"))
        optimizer.load_state_dict(torch.load(args.checkpoint_path+"/optim_"+args.epoch+".pt"))
    train(args, model, optimizer)
