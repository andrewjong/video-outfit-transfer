import argparse
from datetime import datetime
import os
import numpy as np
import tqdm
import math
import sys
import json

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from utils.save import save_models

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", help="path to data to generate training")
parser.add_argument(
    "-e", "--experiment", default="default_experiment", help="name of the experiment"
)
parser.add_argument(
    "-o",
    "--out_dir",
    default=os.path.join("output", "wgan_gp"),
    help="Output folder path",
)
parser.add_argument(
    "--save_dir",
    default=os.path.join("models", "wgan_gp"),
    help="Where to store saved model weights",
)
parser.add_argument(
    "--n_epochs", type=int, default=200, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
)
parser.add_argument(
    "--img_size", type=int, default=512, help="size of each image dimension"
)
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--n_critic",
    type=int,
    default=5,
    help="number of training steps for discriminator per iter",
)
parser.add_argument(
    "--clip_value",
    type=float,
    default=0.01,
    help="lower and upper clip value for disc. weights",
)
parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty")
parser.add_argument(
    "--sample_interval", type=int, default=400, help="interval betwen image samples"
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=10,
    help="epoch interval between model checkpoints",
)
args = parser.parse_args()
print(args)


#######################
# Make output folders
#######################
OUT_DIR = (
    os.path.join(args.out_dir, args.experiment) if args.experiment else args.out_dir
)
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_DIR = (
    os.path.join(args.save_dir, args.experiment) if args.experiment else args.save_dir
)
os.makedirs(os.path.join(MODEL_DIR, "logs"), exist_ok=True)

# Save arguments used
with open(os.path.join(MODEL_DIR, "logs", "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)
# Write progress header
logfile = os.path.join(MODEL_DIR, "train.csv")
with open(logfile, "w") as f:
    f.write("epoch,batch,d_loss,g_loss\n")


def log_progress(*args):
    with open(logfile, "a") as f:
        f.write(",".join((str(a) for a in args)) + "\n")


img_shape = (args.channels, args.img_size, args.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        # print("In gan.forward(), zshape=", z.shape)
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        # print("In gan.forward(): img_shape=", img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()


dataloader = torch.utils.data.DataLoader(
    ImageFolder(
        args.dataset,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(img_shape[1]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
for epoch in tqdm.trange(args.n_epochs, unit="epoch"):
    pbar = tqdm.tqdm(dataloader, unit="batch", leave=False)
    for i, (imgs, _) in enumerate(pbar):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_imgs.data, fake_imgs.data
        )
        # Adversarial loss
        d_loss = (
            -torch.mean(real_validity)
            + torch.mean(fake_validity)
            + args.lambda_gp * gradient_penalty
        )

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % args.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            log_progress(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                i,
                d_loss.item(),
                g_loss.item(),
            )
            pbar.set_description(
                f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}"
            )

            if batches_done % args.sample_interval == 0:
                save_image(
                    fake_imgs.data[:25],
                    os.path.join(OUT_DIR, f"{batches_done}.png"),
                    nrow=5,
                    normalize=True,
                )

            batches_done += args.n_critic

    if epoch % args.checkpoint_interval == 0:
        save_models(
            MODEL_DIR,
            epoch,
            batches_done,
            generator=generator,
            discriminator=discriminator,
        )
