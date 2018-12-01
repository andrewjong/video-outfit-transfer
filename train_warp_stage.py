import argparse

import torch
import torch.utils.data as data
from ignite.engine import Engine

from src.datasets import WarpDataset
from src.loss import PerPixelCrossEntropyLoss
from src.nets import Discriminator
from src.warping_module import WarpingModule

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
parser.add_argument("--cloth_dir")
parser.add_argument("--body_dir")
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--b1", type=float)
parser.add_argument("--b2", type=float)
# no idea what the adversarial weight should be. not mentioned in the paper
parser.add_argument("--adversarial_weight", type=float, default=0.1)
parser.add_argument("--n_cpu", type=int, default=4)
parser.add_argument("--channels", default=19)
parser.add_argument("--img_size", default="960x540")
parser.add_argument("--sample_interval", default=1000)

args = parser.parse_args()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


dataset = WarpDataset(clothing_seg_dir=args.cloth_dir, body_seg_dir=args.body_dir)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, num_workers=args.n_cpu
)


# Loss Warp function
reconstruction_loss = PerPixelCrossEntropyLoss()
adversarial_loss = torch.nn.BCELoss()
# later: warp_loss = reconstruction_loss(i, t) + adversarial_loss(d, c)
# warp_loss.backward()

generator = WarpingModule()
discriminator = Discriminator()


optimizer_g = torch.optim.Adam(
    generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
)
optimizer_d = torch.optim.Adam(
    discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    reconstruction_loss.cuda()
    adversarial_loss.cuda()


def step(engine: Engine):
    """
    A step in the
    :param engine:
    :return:
    """


def main():
    pass


if __name__ == "__main__":

    main()
