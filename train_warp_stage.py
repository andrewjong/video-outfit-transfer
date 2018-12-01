"""
Code modified from PyTorch-GAN:
https://github.com/eriklindernoren/PyTorch-GAN/
"""
import argparse
import datetime
import os
import sys
import time

import torch

# from torchvision.utils import save_image
from tqdm import tqdm

import config
from src.datasets import WarpDataset
from src.loss import PerPixelCrossEntropyLoss
from src.nets import Discriminator, weights_init_normal
from src.warp_module import WarpModule

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Train the warp stage.",
)
parser.add_argument(
    "--body_dir",
    default=config.ANDREW_BODY_SEG,
    help="Path to folder containing body segmentation images (*.png or *.jpg)",
)
parser.add_argument(
    "--clothing_dir",
    default=config.ANDREW_CLOTHING_SEG,
    help="Path to folder containing clothing segmentation images (*.npy)",
)
parser.add_argument(
    "--val_clothing_dir",
    default=config.TINA_CLOTHING_SEG,
    help="Path to folder for validation",
)
parser.add_argument(
    "--dataset_name", default="", help="Name the dataset for output path"
)
parser.add_argument(
    "--out_dir", default=os.path.join("output", "warp_stage"), help="Output folder path"
)
parser.add_argument(
    "--save_dir",
    default=os.path.join("models", "warp_stage"),
    help="Where to store saved model weights",
)
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument(
    "--n_epochs", type=int, default=200, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument(
    "--adversarial_weight",
    type=float,
    default=0.2,
    help="Factor to scale adversarial loss contribution to warp loss total",
)
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
    "--decay_epoch", type=int, default=100, help="epoch from which to start lr decay"
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    help="Probability of dropout on latent space layers",
)
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument(
    "--clothing_channels",
    type=int,
    default=19,
    help="number of channels in the clothing segmentation probability maps",
)
parser.add_argument(
    "--sample_interval",
    type=int,
    default=500,
    help="interval between sampling of images from generators",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=-1,
    help="interval between model checkpoints",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument("--gpu", default="cuda:0", help="Which GPU to train on")
args = parser.parse_args()
print(args)

#######################
# GPU setup
#######################
cuda = True if torch.cuda.is_available() else False
if cuda:
    torch.cuda.set_device(args.gpu)

#######################
# Make output folders
#######################
OUT_DIR = (
    os.path.join(args.out_dir, args.dataset_name) if args.dataset_name else args.out_dir
)
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_DIR = (
    os.path.join(args.save_dir, args.dataset_name)
    if args.dataset_name
    else args.save_dir
)
os.makedirs(MODEL_DIR, exist_ok=True)

###############################
# Model, loss, optimizer setup
###############################

# Loss functions
criterion_GAN = torch.nn.BCELoss()

criterion_pixelwise = PerPixelCrossEntropyLoss()

# Initialize generator and discriminator
generator = WarpModule(cloth_channels=args.clothing_channels, dropout=args.dropout)
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

# only load weights if not first epoch
if args.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(
        torch.load(
            os.path.join(
                args.save_dir, args.dataset_name, f"generator_{args.epoch}.pth"
            )
        )
    )
    discriminator.load_state_dict(
        torch.load(
            os.path.join(
                args.save_dir, args.dataset_name, f"discriminator_{args.epoch}.pth"
            )
        )
    )
else:  # if first epoch,
    # initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
)


###############################
# Datasets and loaders
###############################
# TODO: dataloaders
warp_dataset = WarpDataset(
    body_seg_dir=args.body_dir,
    clothing_seg_dir=args.clothing_dir,
    crop_bounds=config.CROP_BOUNDS,
)
val_dataset = WarpDataset(
    body_seg_dir=args.body_dir,
    clothing_seg_dir=args.val_clothing_dir,
    crop_bounds=config.CROP_BOUNDS,
)
dataloader = torch.utils.data.DataLoader(
    warp_dataset, batch_size=args.batch_size, num_workers=args.n_cpu
)
dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, num_workers=args.n_cpu
)

# def sample_images(batches_done):
#     """Saves a generated sample from the validation set"""
#     imgs = next(iter(val_dataloader))
#     real_A = Variable(imgs["B"].type(Tensor))
#     real_B = Variable(imgs["A"].type(Tensor))
#     fake_B = generator(real_A)
#     img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
#     save_image(
#         img_sample,
#         "images/%s/%s.png" % (args.dataset_name, batches_done),
#         nrow=5,
#         normalize=True,
#     )


###############################
# Training
###############################

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

start_time = time.time()

for epoch in tqdm(
    range(args.epoch, args.n_epochs), desc="Train progress", units="epoch"
):
    batch = tqdm(dataloader, units="batch")
    for bodys, inputs, targets in batch:
        if cuda:
            bodys = bodys.cuda()
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Adversarial ground truths
        valid_labels = torch.ones(targets.shape[0], dtype=Tensor)
        fake_labels = torch.zeros(targets.shape[0], dtype=Tensor)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        gen_fakes = generator(body=bodys, cloth=inputs)
        pred_fake = discriminator(gen_fakes, bodys)
        loss_GAN = criterion_GAN(pred_fake, valid_labels)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(gen_fakes, targets)

        # Total loss
        loss_G = loss_GAN + args.adversarial_weight * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(targets, bodys)
        loss_real = criterion_GAN(pred_real, valid_labels)

        # Fake loss
        pred_fake = discriminator(gen_fakes.detach(), bodys)
        loss_fake = criterion_GAN(pred_fake, fake_labels)

        # Total loss
        loss_D = loss_real + loss_fake

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = args.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - start_time)
        )
        start_time = time.time()

        # Print log
        batch.set_description(
            f"[D loss: {loss_D.item():3f}] [G loss: {loss_G.item():.3f}, "
            + f"pixel: {loss_pixel.item():.3f}, adv: {loss_GAN.item():.3f}]"
        )

        # If at sample interval save image
        # if batches_done % args.sample_interval == 0:
        #     sample_images(batches_done)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(
            generator.state_dict(),
            os.path.join(
                args.save_dir, args.dataset_name, f"generator_{args.epoch}.pth"
            ),
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(
                args.save_dir, args.dataset_name, f"discriminator_{args.epoch}.pth"
            ),
        )
