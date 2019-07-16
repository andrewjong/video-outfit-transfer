"""
Code modified from PyTorch-GAN:
https://github.com/eriklindernoren/PyTorch-GAN/
"""
import argparse
import os
import json
from pprint import pprint
from time import strftime, gmtime

import torch

# from torchvision.utils import save_image
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
from utils.save import save_models

import config
from src.datasets import WarpDataset
from src.loss import PerPixelCrossEntropyLoss
from src.nets import Discriminator, weights_init_normal
from src.warp_module import WarpModule
from utils.decode_labels import decode_cloth_labels

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
    "--cloth_dir",
    default=config.ANDREW_CLOTHING_SEG,
    help="Path to folder containing cloth segmentation images (*.png or *.jpg)",
)
parser.add_argument(
    "-e", "--experiment", default=None, help="Name the dataset for output path. Defaults to current GMT time."
)
parser.add_argument(
    "-o",
    "--out_dir",
    default=os.path.join("output", "warp_stage"),
    help="Output folder path",
)
parser.add_argument(
    "--save_dir",
    default=os.path.join("checkpoints", "warp_stage"),
    help="Where to store saved model weights",
)
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument(
    "--n_epochs", type=int, default=20, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument(
    "--adversarial_weight",
    type=float,
    default=0.2,
    help="Factor to scale adversarial loss contribution to warp loss total",
)
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
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
    "--cloth_channels",
    type=int,
    default=config.cloth_channels,
    help="number of channels in the cloth segmentation probability maps",
)
parser.add_argument(
    "--sample_interval",
    type=int,
    default=500,
    help="batch interval between sampling of images from generators. --val_dir must be set for this argument to work!",
)
parser.add_argument(
    "--val_dir",
    help="Path to cloth folder for validation. Use with samping with the --sample_interval argument.",
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

##################################################################
# new for wgan
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--clamp_lower', type=float, default=-0.05)
parser.add_argument('--clamp_upper', type=float, default=0.05)

##################################################################
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
# use time if no expeirment passed
if not args.experiment:
    time_str = strftime("%Y-%m-%d_%H-%M_%S", gmtime())
    args.experiment = time_str

OUT_DIR = os.path.join(args.out_dir, args.experiment)
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_DIR = (
    os.path.join(args.save_dir, args.experiment)
    if args.experiment
    else args.save_dir
)
os.makedirs(os.path.join(MODEL_DIR, "logs"), exist_ok=True)

# Save arguments used
with open(os.path.join(MODEL_DIR, "logs", "args.json"), "w") as f:
    json.dump(argparse_dict, f, indent=4)
# Write progress header
with open(os.path.join(MODEL_DIR, "logs", "train_log.csv"), "w") as f:
    f.write("step,loss_D,loss_G,loss_pixel,loss_adversarial\n")


###############################
# Model, loss, optimizer setup
###############################

# Loss functions
criterion_labels = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = WarpModule(cloth_channels=args.cloth_channels, dropout=args.dropout)
# cloth channels + RGB
discriminator = Discriminator(in_channels=args.cloth_channels + 3, img_size=args.img_height)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_labels.cuda()

# only load weights if retraining
if args.epoch != 0:
    pass
    # # Load pretrained models
    # generator.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             args.save_dir, args.experiment, f"generator_{args.epoch}.pth"
    #         )
    #     )
    # )
    # discriminator.load_state_dict(
    #     torch.load(
    #         os.path.join(
    #             args.save_dir, args.experiment, f"discriminator_{args.epoch}.pth"
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
        torchvision.transforms.RandomHorizontalFlip(0.3),
        torchvision.transforms.RandomVerticalFlip(0.3),
    )
)
warp_dataset = WarpDataset(
    body_seg_dir=args.body_dir,
    clothing_seg_dir=args.cloth_dir,
    crop_bounds=config.CROP_BOUNDS,
    input_transform=input_transform,
    body_means=config.ANDREW_BODY_SEG_MEAN,
    body_stds=config.ANDREW_BODY_SEG_STD,
)
# if validating, then inference mode is False. if testing new inputs then True.
if args.val_dir:
    val_dataset = WarpDataset(
        body_seg_dir=args.body_dir,
        clothing_seg_dir=args.val_dir,
        crop_bounds=config.CROP_BOUNDS,
        input_transform=input_transform,
        inference_mode=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu
    )


def sample_images(epoch, batches_done, num_images=5):
    """Saves a generated sample from the validation set"""
    bodys, inputs, targets = next(iter(val_dataloader))
    if cuda:
        bodys = bodys.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
    fakes = generator(bodys, inputs)

    # scale and decode cloth labels
    rgb_bodys = ((bodys - bodys.min()) * 255 / (bodys.max() - bodys.min())).byte().cpu()[:num_images]
    decoded_inputs = decode_cloth_labels(inputs.detach().cpu())[:num_images]
    decoded_fakes = decode_cloth_labels(fakes.detach().cpu())[:num_images]
    decoded_targets = decode_cloth_labels(targets.detach().cpu())[:num_images]

    img_sample = torch.cat((rgb_bodys, decoded_inputs, decoded_fakes, decoded_targets), dim=-2)

    save_image(
        img_sample,
        os.path.join(OUT_DIR, f"{epoch:02d}_{batches_done:05d}.png"),
        nrow=args.batch_size,
        normalize=True,
        scale_each=True # normalize each bodys/inputs/fakes/targets separately
    )



###############################
# Training
###############################


gen_data = torch.utils.data.DataLoader(
    warp_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu
)
# a separate dataloader for training critic
critic_data = torch.utils.data.DataLoader(
    warp_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True,
)
critic_iter = iter(critic_data)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

one = torch.FloatTensor([1]).cuda()
mone = one * -1

gen_iterations = 0
for epoch in tqdm(
    range(args.epoch, args.n_epochs), desc="Completed Epochs", unit="epoch"
):
    gen_batch = tqdm(gen_data, unit="batch")
    for i, (bodys, inputs, targets) in enumerate(gen_batch):
    
        ############################
        # (1) Update D network
        ###########################
        discriminator.train()
        generator.eval()

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 10
        else:
            Diters = args.Diters
        
        for critic_step in range(Diters):

            # clamp parameters to a cube
            for p in discriminator.parameters():
                p.data.clamp_(args.clamp_lower, args.clamp_upper)

            discriminator.zero_grad()
            try:
                bodys, inputs, targets = critic_iter.next()
            except StopIteration:
                print('Reloading critic data...',end='')
                critic_data = torch.utils.data.DataLoader(
                    warp_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True,
                )
                print('done')
                critic_iter = iter(critic_data)
                bodys, inputs, targets = critic_iter.next()
                
            bodys = bodys.cuda()
            inputs = inputs.cuda()
            targets = targets.cuda()

            i += 1

            # train with real
            errD_real = discriminator(targets, bodys).mean(0)
            errD_real.backward(one)

            # train with fake
            with torch.no_grad(): # freeze the generator
                fakes = generator(body=bodys, cloth=inputs)

            errD_fake = discriminator(fakes, bodys).mean(0)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizer_D.step()
                
            print(f'[{critic_step}/{Diters}] Loss_D: {errD.item()} Loss_D_real: {errD_real.item()} Loss_D_fake: {errD_fake.item()}')

                
        ############################
        # (2) Update G network
        ###########################
        bodys = bodys.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        discriminator.eval()
        generator.train()
        generator.zero_grad()

        fakes = generator(body=bodys, cloth=inputs)
        with torch.no_grad():
            loss_adv = discriminator(fakes, bodys).mean(0)
        # Pixel-wise loss
        loss_pixel = criterion_labels(fakes, torch.argmax(targets, dim=1))
        loss_warp = loss_pixel + args.adversarial_weight * loss_adv
        loss_warp.backward(one)
        optimizer_G.step()
        gen_iterations += 1
                
        ###############################################
        
#         gen_batch.set_description(
#             f"[G loss: {loss_warp.item():.3f}, pixel: {loss_pixel.item():.3f}, adv: {loss_adv.item():.3f}]"
#         )
        print(f"\n[G loss: {loss_warp.item():.3f}, pixel: {loss_pixel.item():.3f}, adv: {loss_adv.item():.3f}]\n")

        # Save log to file
        batches_done = epoch * len(gen_data) + i
#         with open(os.path.join(MODEL_DIR, "logs", "train_log.csv"), "a+") as f:
#             f.write(
#                 ",".join(
#                     (
#                         str(batches_done),
#                         str(loss_D.item()),
#                         str(loss_warp.item()),
#                         str(loss_pixel.item()),
#                         str(loss_adv.item()),
#                     )
#                 )
#             )
#             f.write("\n")

        # ------------------------------
        #  Sample images and save model
        # ------------------------------
        # If at sample interval save image
        if args.val_dir and batches_done % args.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                sample_images(epoch, batches_done)
            generator.train()

        if (
            args.checkpoint_interval != -1
            and batches_done % args.checkpoint_interval == 0
        ):
            save_models(
                MODEL_DIR,
                epoch,
                batches_done,
                generator=generator,
                discriminator=discriminator,
            )

        # ------------------------------
        # End train if starts to destabilize
        # ------------------------------
        # numbers determined experimentally
#         if loss_D < 0.005 and loss_warp > 8:
#             print(
#                 "Loss_D is less than 0.05 and loss_warp > 3!",
#                 "Saving models and ending train to prevent destabilization.",
#             )
#             sample_images(-1, -1)
#             save_models(
#                 MODEL_DIR, -1, -1, generator=generator, discriminator=discriminator
#             )
#             break
    else:
        continue
    break
