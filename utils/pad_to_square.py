import glob
import os
from tqdm import tqdm
import argparse
import cv2
from PIL import ImageOps
import numpy as np


def pad_to_square(image: np.ndarray):
    new_size = max(image.shape)

    h, w = image.shape[:2]

    pad_h = (new_size - h) // 2
    pad_w = (new_size - w) // 2

    padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    return padded


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pad an image to square, then resize to the desired size")
    parser.add_argument("image_folder", help="path to images root folder")
    parser.add_argument("output_folder", help="output folder path")
    parser.add_argument("--size", type=int, default=None, help="final output size to resize to, if no argument passed then keeps the size")

    args = parser.parse_args()

    images = glob.glob(args.image_folder + "**/*.jpg", recursive=True)
    for img_path in tqdm(images):
        nodir = img_path.replace(args.image_folder, "").lstrip(os.path.sep)
        img = cv2.imread(img_path)
        img = pad_to_square(img)
        if args.size is not None:
            img = cv2.resize(img, (args.size, args.size))

        out_path = os.path.join(args.output_folder, nodir)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
