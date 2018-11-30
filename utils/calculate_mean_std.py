from src.datasets import WarpDataset
import numpy as np
import os
from PIL import Image
from config import (
    ANDREW_CLOTHING_SEG,
    ANDREW_BODY_SEG,
    TINA_CLOTHING_SEG,
    TINA_BODY_SEG,
)

# DEPRECATED, DO NOT USE THIS FILE

def get_file_loader(ext: str):
    """
    Get the appropriate loader to convert a file into a numpy array.

    :param ext: extension
    :return:
    """
    if ext is ".npy":
        load_op = np.load
    elif ext is ".png" or ext is ".jpg":
        # run two functions sequentially
        def compose(func_f, func_g):
            return lambda arg: func_g(func_f(arg))

        load_op = compose(Image.open, np.array)
    else:
        raise ValueError(
            f"ext argument must be either '.npy', '.jpg', or '.png'. got {ext}"
        )

    return load_op


for dataset in (ANDREW_BODY_SEG, TINA_BODY_SEG, ANDREW_CLOTHING_SEG, TINA_CLOTHING_SEG):
    # calculate mean and std
    files = os.listdir(dataset)
    first = files[0]

    ext = os.path.splitext(first)[-1]
    load_op = get_file_loader(ext)

    shape = load_op(first).shape
    accumulation = np.zeros(shape)

    for f in files:
        x = load_op(f)
        accumulation += x

    # broadcasting, to divide each channel by the length
    axes = tuple(range(len(shape)))
    # calculate the stats
    means = np.mean(accumulation, axis=axes)
    stds = np.std(accumulation, axis=axes)

    d_name = os.path.basename(dataset)
    print(f"Dataset {d_name}: means={means} stds={stds}")
