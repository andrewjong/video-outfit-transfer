import argparse
import os
from pprint import pprint

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import config
from src.datasets import TextureDataset
from src.texture_module import TextureModule

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Inference the warp stage.",
)
parser.add_argument("-m", "--model", help="Directory to the model *.pth generator file")
parser.add_argument(
    "-t",
    "--texture_dir",
    default=config.ANDREW_BODY_SEG,
    help="Path to folder containing texture images (*.png or *.jpg)",
)
parser.add_argument(
    "-r",
    "--roidb",
    default=config.ANDREW_ROIS,
    help="Path to csv file containing ROIs",
)
parser.add_argument(
    "-c",
    "--clothing_dir",
    default=config.ANDREW_CLOTHING_SEG,
    help="Path to folder containing posed clothing segmentation images (*.png or "
    "*.jpg)",
)
parser.add_argument(
    "-o",
    "--out_dir",
    default=os.path.join("output", "texture_stage", "inference"),
    help="Output folder path",
)
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument("--gpu", default=0, type=int, help="Set which GPU to run on")
args = parser.parse_args()
argparse_dict = vars(args)
pprint(argparse_dict)

#######################
# GPU setup
#######################
cuda = True if torch.cuda.is_available() else False
# if cuda:
#     torch.cuda.set_device(args.gpu)

# Initialize generator and discriminator
generator = TextureModule()
generator.load_state_dict(
    torch.load(args.model, map_location=lambda storage, loc: storage)
)
generator.eval()
if cuda:
    generator = generator.cuda()

print("Model loaded!")

###############################
# Datasets and loaders
###############################
texture_dataset = TextureDataset(
    args.texture_dir, args.rois_db, args.clothing_dir, crop_bounds=config.CROP_BOUNDS
)
dataloader = torch.utils.data.DataLoader(
    texture_dataset, batch_size=args.batch_size, num_workers=args.n_cpu
)

os.makedirs(args.out_dir, exist_ok=True)

frame_num = 0
for textures, rois, clothing in tqdm(dataloader):
    if cuda:
        textures = textures.cuda()
        rois = rois.cuda()
        clothing = clothing.cuda()

    gen_fakes = generator(textures, rois, clothing)
    for i in range(gen_fakes.shape[0]):
        img = gen_fakes[i, :, :, :]
        fname = os.path.join(args.out_dir, f"{frame_num:04d}.png")
        save_image(img, fname, nrow=1, normalize=True)
        frame_num += 1
