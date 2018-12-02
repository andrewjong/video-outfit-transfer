import os
import random
from typing import Set, List, Tuple
import torchvision

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


to_tensor = torchvision.transforms.ToTensor()


class WarpDataset(Dataset):
    def __init__(
        self,
        body_seg_dir: str,
        clothing_seg_dir: str,
        crop_bounds: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        min_offset: int = 100,
        random_seed=None,
        input_transform=None,
        body_means=None,
        body_stds=None,
    ):
        """
        Warp dataset for the warping module of SwapNet. All files in the dataset must be
        from the same subject (and ideally with the same background)

        :param clothing_seg_dir: path to directory containing clothing segmentation
        .npy files
        :param body_seg_dir: path to directory containing body segmentation image files
        :param min_offset: minimum offset to select a random other clothing
        segmentation. (default: 50; unless min_offset < number of clothing files,
        then 0)
        :param random_seed: seed for getting a random clothing seg image
        :param input_transform: transform for the random drawn clothing segmentation image.
        Note, the transform must be able to operate on a HxWx19-channel tensor
        """
        self.body_seg_dir = body_seg_dir
        # A set of file names, mapping to RGB images
        # we choose a set for fast lookups
        self.body_seg_files_set: Set[str] = set(os.listdir(body_seg_dir))
        # file extension of the body seg images. probably .png or .jpg
        first_bs = next(iter(self.body_seg_files_set))
        self.body_seg_ext = os.path.splitext(first_bs)[-1]

        self.clothing_seg_dir = clothing_seg_dir
        # list of file names, mapping to npy arrays
        self.clothing_seg_files: List[str] = os.listdir(clothing_seg_dir)

        self.crop_bounds = crop_bounds

        self.min_offset = min_offset if len(self.clothing_seg_files) < min_offset else 0
        if random_seed:
            random.seed(random_seed)
        self.random_seed = random_seed
        self.input_transform = input_transform

        # transforms for RGB images
        body_transforms = [torchvision.transforms.ToTensor()]
        if body_means and body_stds:
            body_transforms.append(
                torchvision.transforms.Normalize(body_means, body_stds)
            )
        self.body_transforms = torchvision.transforms.Compose(body_transforms)

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

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Strategy:
            Get a target clothing segmentation (.npy). Get the matching body
            segmentation (.png)
            Choose a random starting clothing segmentation from a different frame
            (.npy), and perform data augmention on each channel of that frame

        :param index:
        :return: body segmentation, input clothing segmentation, target clothing
        segmentation
        """
        # Load as np arrays
        target_cs_file = self.clothing_seg_files[index]
        target_cs_img = Image.open(os.path.join(self.clothing_seg_dir, target_cs_file))

        input_cs_file = self._get_random_clothing_seg(index)
        input_cs_img = Image.open(os.path.join(self.clothing_seg_dir, input_cs_file))

        # the body segmentation that corresponds to the target
        body_seg_file = self._get_matching_body_seg_file(target_cs_file)
        body_seg_img = Image.open(os.path.join(self.body_seg_dir, body_seg_file))

        # apply the transformation if desired
        if self.input_transform:
            input_cs_img = self.input_transform(input_cs_img)

        # convert to PT tensors and return
        # TODO: normalize the tensors
        body_s = self.body_transforms(body_seg_img)
        input_cs = to_tensor(input_cs_img)
        target_cs = to_tensor(target_cs_img)

        # crop to the proper image size
        if self.crop_bounds is not None:
            body_s = self._crop(body_s)
            input_cs = self._crop(input_cs)
            target_cs = self._crop(target_cs)

        return body_s, input_cs, target_cs

    def _crop(self, tensor: Tensor):
        (h_min, hmax), (w_min, w_max) = self.crop_bounds
        return tensor[:, h_min:hmax, w_min:w_max]


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
