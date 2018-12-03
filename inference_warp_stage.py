import argparse
import os
from pprint import pprint

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import config
from src.datasets import WarpDataset
from src.warp_module import WarpModule

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Inference the warp stage.",
)
parser.add_argument("-m", "--model", help="Directory to the model *.pth generator file")
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
    "-o",
    "--out_dir",
    default=os.path.join("output", "warp_stage", "inference"),
    help="Output folder path",
)
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
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
# if cuda:
#     torch.cuda.set_device(args.gpu)

# Initialize generator and discriminator
generator = WarpModule(cloth_channels=3, dropout=0)
generator.load_state_dict(
    torch.load(args.model, map_location=lambda storage, loc: storage)
)
if cuda:
    generator = generator.cuda()

print("Model loaded!")

###############################
# Datasets and loaders
###############################
warp_dataset = WarpDataset(
    body_seg_dir=args.body_dir,
    clothing_seg_dir=args.clothing_dir,
    crop_bounds=config.CROP_BOUNDS,
    inference_mode=True,
)
dataloader = torch.utils.data.DataLoader(
    warp_dataset, batch_size=args.batch_size, num_workers=args.n_cpu
)

frame_num = 0
for bodys, inputs, _ in tqdm(dataloader):
    if cuda:
        bodys = bodys.cuda()
        inputs = inputs.cuda()

    gen_fakes = generator(body=bodys, cloth=inputs)
    for i in range(gen_fakes.shape[0]):
        img = gen_fakes[i, :, :, :]
        fname = os.path.join(args.out_dir, f"{frame_num:04d}.png")
        save_image(img, fname, nrow=1, normalize=True)
        frame_num += 1
