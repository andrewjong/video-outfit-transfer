import os
import random
from typing import Set, List, Tuple

import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset


class WarpDataset(Dataset):
    def __init__(
        self,
        clothing_seg_dir: str,
        body_seg_dir: str,
        min_offset: int = 50,
        random_seed=None,
        transform=None,
    ):
        """
        Warp dataset for the warping module of SwapNet. All files in the dataset must be
        from the same subject (and ideally with the same background)


        Strategy:
            Get a target clothing segmentation (.npy). Get the matching body
            segmentation (.png)
            Choose a random starting clothing segmentation at a different frame (.npy),
            and perform data augmention on each channel of that frame


        :param clothing_seg_dir: path to directory containing clothing segmentation
        .npy files
        :param body_seg_dir: path to directory containing body segmentation image files
        :param min_offset: minimum offset to select a random other clothing
        segmentation (default: 50)
        :param random_seed: seed for getting a random clothing seg image
        :param transform: transform for the random drawn clothing segmentation image.
        Note, the transform must be able to operate on a HxWx19-channel tensor
        """
        self.clothing_seg_dir = clothing_seg_dir
        # list of file names, mapping to npy arrays
        self.clothing_seg_files: List[str] = os.listdir(clothing_seg_dir)

        self.body_seg_dir = body_seg_dir
        # A set of file names, mapping to RGB images
        # we choose a set for fast lookups
        self.body_seg_files_set: Set[str] = set(os.listdir(body_seg_dir))

        # file extension of the body seg images. probably .png or .jpg
        first_bs = next(iter(self.body_seg_files_set))
        self.body_seg_ext = os.path.splitext(first_bs)[-1]

        self.min_offset = min_offset
        self.random_seed = random_seed
        self.transform = transform

    def __len__(self):
        """
        Get the length of usable images
        :return: length of the image
        """
        # it's possible one file list will not be complete, i.e. missing
        # corresponding files. if that's the case, the number of images we can use is
        # the length of the smaller list
        smaller_length = min(len(self.clothing_seg_files), len(self.body_seg_files_set))
        return smaller_length

    def _get_matching_body_seg_file(self, clothing_seg_fname: str):
        """
        For a given clothing segmentation file, get the matching body segmentation
        file that corresponds to it.
        :param clothing_seg_fname:
        :return:
        :raises ValueError: if no corresponding body segmentation image found.
        """
        base_fname = os.path.basename(clothing_seg_fname)
        fname_no_extension = os.path.splitext(base_fname)[0]
        body_seg_fname = fname_no_extension + self.body_seg_ext

        if body_seg_fname in self.body_seg_files_set:
            return body_seg_fname
        else:
            raise ValueError(
                "No corresponding body segmentation image found. "
                "Could not find: " + body_seg_fname
            )

    def _get_random_clothing_seg(self, index):
        """
        Note, this implementation isn't perfect, but should be good enough for now (
        we're on a deadline).

        Unaccounted corner cases: index == 0 or index == max-index
        :param index:
        :return:
        """
        min_thresh = index - self.min_offset
        max_thresh = index + self.min_offset + 1  # + 1 so that
        # make sure we're not out-of-bounds
        if min_thresh < 0:
            min_thresh = 0
        if max_thresh >= len(self.clothing_seg_files):
            max_thresh = len(self.clothing_seg_files) - 1

        # our valid set
        valid_choices = (
            self.clothing_seg_files[:min_thresh] + self.clothing_seg_files[max_thresh:]
        )

        return random.choice(valid_choices)

    def __getitem__(self, index) -> Tuple:
        """
        TODO: transform all outputs to Tensors

        :param index:
        :return: ndarray, ndarray, PIL.Image
        """
        # Load as np arrays
        target_cs_file = self.clothing_seg_files[index]
        target_cs_ndarray = np.load(os.path.join(self.clothing_seg_dir, target_cs_file))
        # the files are shaped (1, w, h, c). we shouldn't have that extra dimension in front
        target_cs_ndarray = np.squeeze(target_cs_ndarray, 0)

        other_cs_file = self._get_random_clothing_seg(index)
        other_cs_ndarray = np.load(os.path.join(self.clothing_seg_dir, other_cs_file))
        other_cs_ndarray = np.squeeze(other_cs_ndarray, 0)
        # apply the transformation if desired
        if self.transform:
            other_cs_ndarray = self.transform(other_cs_ndarray)

        # the body segmentation that corresponds to the target
        body_seg_file = self._get_matching_body_seg_file(target_cs_file)
        body_seg_img = Image.open(os.path.join(self.body_seg_dir, body_seg_file))

        to_tensor = ToTensor()
        return (
            to_tensor(target_cs_ndarray),
            to_tensor(other_cs_ndarray),
            to_tensor(body_seg_img),
        )


class TextureDataset(Dataset):
    def __init__(self) -> None:
        """



        Strategy:
            Get a target photo (.png). Get the matching clothing
            segmentation (.npy).
        """
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
