"""
Code modified from PyTorch-GAN:
https://github.com/eriklindernoren/PyTorch-GAN/
"""
import argparse
import os
import json
from pprint import pprint

import torch

# from torchvision.utils import save_image
import torchvision
from torchvision.utils import save_image
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
    "-b",
    "--body_dir",
    default=config.ANDREW_BODY_SEG,
    help="Path to folder containing body segmentation images (*.png or *.jpg)",
)
parser.add_argument(
    "-c",
    "--clothing_dir",
    default=config.ANDREW_CLOTHING_SEG,
    help="Path to folder containing clothing segmentation images (*.png or *.jpg)",
)
parser.add_argument(
    "-d", "--dataset_name", default="", help="Name the dataset for output path"
)
parser.add_argument(
    "-o", "--out_dir",
    default=os.path.join("output", "warp_stage"),
    help="Output folder path",
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
    default=3,
    help="number of channels in the clothing segmentation probability maps",
)
parser.add_argument(
    "--sample_interval",
    type=int,
    default=500,
    help="interval between sampling of images from generators. --val_dir must be set for this argument to work!",
)
parser.add_argument(
    "--val_dir",
    help="Path to folder for validation. Use with samping with the --sample_interval argument.",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=10,
    help="interval between model checkpoints",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument("--gpu", default=0, type=int, help="Set which GPU to train on")
args = parser.parse_args()
argparse_dict = vars(args)
pprint(argparse_dict)

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
os.makedirs(os.path.join(MODEL_DIR, "logs"), exist_ok=True)

# Save arguments used
with open(os.path.join(MODEL_DIR, "logs", "args.json"), "w") as f:
    json.dump(argparse_dict, f, indent=4)
# Write progress header
with open(os.path.join(MODEL_DIR, "logs", "train_log.csv"), "w") as f:
    f.write("step,loss_D,loss_G,loss_pixel,loss_adversarial")


###############################
# Model, loss, optimizer setup
###############################

# Loss functions
criterion_GAN = torch.nn.BCELoss()
criterion_pixelwise = PerPixelCrossEntropyLoss()

# Initialize generator and discriminator
generator = WarpModule(cloth_channels=args.clothing_channels, dropout=args.dropout)
# clothing channels + RGB
discriminator = Discriminator(in_channels=args.clothing_channels + 3)

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
input_transform = torchvision.transforms.Compose(
    (
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(
            degrees=20, translate=(0.4, 0.4), scale=(0.75, 1.25), shear=(-10, 10)
        ),
    )
)
warp_dataset = WarpDataset(
    body_seg_dir=args.body_dir,
    clothing_seg_dir=args.clothing_dir,
    crop_bounds=config.CROP_BOUNDS,
    input_transform=input_transform,
)
dataloader = torch.utils.data.DataLoader(
    warp_dataset, batch_size=args.batch_size, num_workers=args.n_cpu
)

if args.val_dir:
    val_dataset = WarpDataset(
        body_seg_dir=args.body_dir,
        clothing_seg_dir=args.val_dir,
        crop_bounds=config.CROP_BOUNDS,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.n_cpu
    )


os.makedirs(os.path.join("images", "warp_module", args.dataset_name), exist_ok=True) 
def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    bodys, inputs, targets = next(iter(val_dataloader))
    if cuda:
        bodys = bodys.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
    fakes = generator(bodys, inputs)
    img_sample = torch.cat((bodys.data, inputs.data, fakes.data, targets.data), -2)
    save_image(
        img_sample,
        os.path.join("images", "warp_module", args.dataset_name, f"{batches_done}.png"),
        nrow=5,
        normalize=True,
    )


###############################
# Training
###############################

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in tqdm(
    range(args.epoch, args.n_epochs), desc="Completed Epochs", unit="epoch"
):
    batch = tqdm(dataloader, unit="batch")
    for i, (bodys, inputs, targets) in enumerate(batch):
        if cuda:
            bodys = bodys.cuda()
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Adversarial ground truths
        valid_labels = torch.ones(targets.shape[0]).type(Tensor)
        fake_labels = torch.zeros(targets.shape[0]).type(Tensor)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        gen_fakes = generator(body=bodys, cloth=inputs)
        pred_fake = discriminator(gen_fakes, bodys)
        loss_adv = criterion_GAN(pred_fake, valid_labels)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(gen_fakes, targets)

        # Total loss
        loss_warp = loss_pixel + args.adversarial_weight * loss_adv

        loss_warp.backward()

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
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        # Print log
        batch.set_description(
            f"[D loss: {loss_D.item():3f}] [G loss: {loss_warp.item():.3f}, "
            + f"pixel: {loss_pixel.item():.3f}, adv: {loss_adv.item():.3f}]"
        )

        # Save to file
        batches_done = epoch * len(dataloader) + i
        with open(os.path.join(MODEL_DIR, "logs", "train_log.csv"), "a+") as f:
            f.write(
                ",".join(
                    (
                        str(batches_done),
                        str(loss_D.item()),
                        str(loss_warp.item()),
                        str(loss_pixel.item()),
                        str(loss_adv.item()),
                    )
                )
            )
            f.write("\n")

        # If at sample interval save image
        if args.val_dir and batches_done % args.sample_interval == 0:
            sample_images(batches_done)

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
