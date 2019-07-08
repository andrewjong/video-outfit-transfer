"""
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre

gist source: https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
"""

import numpy as np
from os import listdir, chdir
from tqdm import tqdm
from os.path import join, isdir
from glob import glob
import cv2
import timeit
import sys

sys.path.append('.')

from config import (
#     ANDREW_CLOTHING_SEG,
#     ANDREW_BODY_SEG,
#     TINA_CLOTHING_SEG,
#     TINA_BODY_SEG,
    BODY_SEG,
)

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
CHANNEL_NUM = 3


def cal_dir_stat(root):
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    im_pths = glob(join(root,"**/*"+".jpg"), recursive=True)
    print(len(im_pths))

    for path in tqdm(im_pths):
        im = cv2.imread(path)  # image in M*N*3 shape, channel in BGR order
        im = im / 255.0
        pixel_num += im.size / CHANNEL_NUM
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    return rgb_mean, rgb_std


# The script assumes that under train_root, there are separate directories for each class
# of training images.
train_root = BODY_SEG
start = timeit.default_timer()
mean, std = cal_dir_stat(train_root)
end = timeit.default_timer()
print("elapsed time: {}".format(end - start))
print("mean:{}\nstd:{}".format(mean, std))
