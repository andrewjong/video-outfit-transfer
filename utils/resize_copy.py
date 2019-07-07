# requires Python 3

import os
import glob
import argparse
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("source", help="where to get files from")
parser.add_argument("dest", help="where to link files to")
parser.add_argument("--filetype", default=".jpg", help="filetype")
parser.add_argument("--levels", type=int, default=1)
parser.add_argument("--sep", default="_")

args = parser.parse_args()

files = glob.glob(args.source + "/**/*" + args.filetype, recursive=True)
files = sorted(files)

os.makedirs(args.dest, exist_ok=True)

for f in tqdm(files):
    head = f
    components = []
    for i in range(args.levels):
        head, tail = os.path.split(head)
        components.append(tail)

    components.reverse()
    base = args.sep.join(components)
    im = Image.open(f)
    im = im.resize((512,512))
    im.save(os.path.join(args.dest, base), "JPEG")

    # os.symlink(os.path.abspath(f), os.path.abspath(os.path.join(args.dest, base)))

