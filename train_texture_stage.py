"""
Code modified from PyTorch-GAN:
https://github.com/eriklindernoren/PyTorch-GAN/
"""
import argparse
import json
import os
from pprint import pprint

import torch

# from torchvision.utils import save_image
import torchvision
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm

import config
from src.datasets import TextureDataset
from src.loss import L1FeatureLoss
from src.loss import MultiLayerFeatureLoss
from src.nets import Discriminator, weights_init_normal
from src.texture_module import TextureModule

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Train the warp stage.",
)
parser.add_argument(
    "-t",
    "--texture_dir",
    default=config.ANDREW_TEXTURE,
    help="Path to folder containing original texture images",
)
parser.add_argument(
    "-r",
    "--rois_db",
    default=config.ANDREW_ROIS,
    help="Path to file containing the rois info",
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
    "-o",
    "--out_dir",
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
    "--n_epochs", type=int, default=20, help="number of epochs of training"
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
    "--texture_channels",
    type=int,
    default=3,
    help="number of channels in the texture images",
)
parser.add_argument(
    "--sample_interval",
    type=int,
    default=500,
    help="batch interval between sampling of images from generators. --val_dir must be set for this argument to work!",
)
parser.add_argument(
    "--val_dir",
    help="Path to folder for validation. Use with samping with the --sample_interval argument.",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=1000,
    help="batch interval between model checkpoints",
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
    f.write(
        "step,loss_D,loss_G,loss_texture,loss_feature_l1,loss_mlf," "loss_adversarial\n"
    )


###############################
# Model, loss, optimizer setup
###############################
feature_extractor = models.vgg19(pretrained=True)

# Loss functions
criterion_GAN = torch.nn.BCELoss()  # binary cross entropy
criterion_l1_feat = L1FeatureLoss(feature_extractor)
criterion_mlf = MultiLayerFeatureLoss(feature_extractor)

# Initialize generator and discriminator
generator = TextureModule(texture_channels=args.texture_channels, dropout=args.dropout)
# clothing channels + RGB
discriminator = Discriminator(in_channels=args.texture_channels + 3)

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion_GAN.cuda()
    criterion_l1_feat.cuda()

# only load weights if retraining
if args.epoch != 0:
    pass
    # # Load pretrained models
    # generator.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             args.save_dir, args.dataset_name, f"generator_{args.epoch}.pth"
    #         )
    #     )
    # )
    # discriminator.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             args.save_dir, args.dataset_name, f"discriminator_{args.epoch}.pth"
    #         )
    #     )
    # )
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
texture_dataset = TextureDataset(
    args.texture_dir, args.rois_db, args.clothing_dir, crop_bounds=config.CROP_BOUNDS
)
dataloader = torch.utils.data.DataLoader(
    texture_dataset, batch_size=args.batch_size, num_workers=args.n_cpu
)

if args.val_dir:
    val_dataset = TextureDataset(
        args.texture_dir, args.rois_db, args.val_dir, config.CROP_BOUNDS
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.n_cpu
    )


def sample_images(epoch, batches_done):
    """Saves a generated sample from the validation set"""
    textures, rois, clothing = next(iter(val_dataloader))
    if cuda:
        textures = textures.cuda()
        rois = rois.cuda()
        clothing = clothing.cuda()
    fakes = generator(textures, rois)
    img_sample = torch.cat((textures.data, rois.data, fakes.data, clothing.data), -2)
    save_image(
        img_sample,
        os.path.join(OUT_DIR, f"{epoch:02d}_{batches_done:05d}.png"),
        nrow=args.batch_size,
        normalize=True,
    )


###############################
# Training
###############################


def save_models(epoch, batches_done):
    # Save model checkpoints
    torch.save(
        generator.state_dict(),
        os.path.join(MODEL_DIR, f"generator_{epoch:02d}_{batches_done:05d}.pth"),
    )
    torch.save(
        discriminator.state_dict(),
        os.path.join(MODEL_DIR, f"discriminator_{epoch:02d}_{batches_done:05d}.pth"),
    )


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Train Loop

for epoch in tqdm(
    range(args.epoch, args.n_epochs), desc="Completed Epochs", unit="epoch"
):
    batch = tqdm(dataloader, unit="batch")
    for i, (textures, rois, cloths) in enumerate(batch):
        if cuda:
            textures = textures.cuda()
            rois = rois.cuda()
            cloths = cloths.cuda()

        # Adversarial ground truths
        valid_labels = torch.ones(cloths.shape[0]).type(Tensor)
        fake_labels = torch.zeros(cloths.shape[0]).type(Tensor)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        gen_fakes = generator(textures, rois, cloths)
        pred_fake = discriminator(gen_fakes, textures)
        loss_adv = criterion_GAN(pred_fake, valid_labels)

        loss_f1 = criterion_l1_feat(gen_fakes, textures)
        loss_mlf = criterion_mlf(gen_fakes, textures)

        # Total loss
        loss_texture = loss_f1 + loss_mlf + loss_adv

        loss_texture.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(gen_fakes, textures)
        loss_real = criterion_GAN(pred_real, valid_labels)

        # Fake loss
        pred_fake = discriminator(gen_fakes.detach(), textures)
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
            f"[D loss: {loss_D.item():3f}] [G loss: {loss_texture.item():.3f}, "
            f"f1: {loss_f1.item():.3f}, mlf: {loss_mlf.item():.3f}, "
            f"adv: {loss_adv.item():.3f}]"
        )

        # Save log to file
        batches_done = epoch * len(dataloader) + i
        with open(os.path.join(MODEL_DIR, "logs", "train_log.csv"), "a+") as f:
            f.write(
                ",".join(
                    (
                        str(batches_done),
                        str(loss_D.item()),
                        str(loss_texture.item()),
                        str(loss_f1.item()),
                        str(loss_mlf.item()),
                        str(loss_adv.item()),
                    )
                )
            )
            f.write("\n")

        # ------------------------------
        #  Sample images and save model
        # ------------------------------
        # If at sample interval save image
        if args.val_dir and batches_done % args.sample_interval == 0:
            sample_images(epoch, batches_done)

        if (
            args.checkpoint_interval != -1
            and batches_done % args.checkpoint_interval == 0
        ):
            save_models(epoch, batches_done)

        # ------------------------------
        # End train if starts to destabilize
        # ------------------------------
        # numbers determined experimentally
        if loss_D < 0.05 or loss_texture > 3:
            print(
                "Loss_D is less than 0.05 or loss_warp > 3!",
                "Saving models and ending train to prevent destabilization.",
            )
            sample_images(-1, -1)
            save_models(-1, -1)
            break
    else:
        continue
    break
