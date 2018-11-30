# Convert to scipy sparse matrices

# Clip low probability values to 0
import os
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import gc

import numpy as np
import argparse


def chunkify(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def clip_files(files):
    arrays = {}

    first = os.path.splitext(files[0])[0]
    last = os.path.splitext(files[-1])[0]

    o_name = first+"-"+last
    if args.out_dir:
        out = os.path.join(args.out_dir, o_name)
    else:
        out = os.path.join(args.data_dir, o_name)

    print("Saving to", out)

    for f in tqdm(files, position=1):
        full = os.path.join(args.data_dir, f)
        a = np.load(full)
        # clip the array
        a = a[a < args.clip]
        f_name = os.path.basename(f)
        arrays[f_name] = a

    np.savez_compressed(out, arrays)
    # free memory
    arrays=None
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir")
    parser.add_argument("-o", "--out_dir", help="Output directory of converted ")
    parser.add_argument(
        "-c",
        "--clip",
        type=float,
        default=0.05,
        help="Clip to 0 if value is below this number",
    )
    parser.add_argument("--num_threads", type=int, default=1)

    args = parser.parse_args()

    all_files = os.listdir(args.data_dir)

    partitions = chunkify(all_files, args.num_threads)

    for p in tqdm(iter(partitions), position=0):
        clip_files(p)

    print("ALL DONE!")
