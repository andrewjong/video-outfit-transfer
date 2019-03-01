import os
import pandas as pd
import random
from typing import Set, List, Tuple
import torchvision
import os.path as op
import torchvision.transforms.functional as t_func

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset

to_tensor = torchvision.transforms.ToTensor()


def crop(tensor: Tensor, crop_bounds):
    """
    Crops a tensor at the given crop bounds.
    :param tensor:
    :param crop_bounds: ((h_min, h_max), (w_min,w_max))
    :return:
    """
    (h_min, hmax), (w_min, w_max) = crop_bounds
    return tensor[:, h_min:hmax, w_min:w_max]


def crop_rois(rois: np.ndarray, crop_bounds):
    # TODO: might have to worry about nan values?
    if crop_bounds is not None:
        rois = rois.copy()
        (hmin, hmax), (wmin, wmax) = crop_bounds
        # clip the x-axis to be within bounds. xmin and xmax index
        xs = rois[:, (1, 2)]
        xs = np.clip(xs, wmin, wmax - 1)
        xs -= xs.min(axis=0)  # translate
        # clip the y-axis to be within bounds. ymin and ymax index
        ys = rois[:, (3, 4)]
        ys = np.clip(ys, hmin, hmax - 1)
        ys -= ys.min(axis=0)  # translate
        # put it back together again
        rois = np.stack((rois[:, 0], xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]))
        # transpose because stack stacked them opposite of what we want
        rois = rois.T
    return rois


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
        inference_mode=False,
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
        self.body_seg_files = sorted(os.listdir(body_seg_dir))
        self.body_seg_files_set: Set[str] = set(self.body_seg_files)
        # file extension of the body seg images. probably .png or .jpg
        first_bs = next(iter(self.body_seg_files_set))
        self.body_seg_ext = op.splitext(first_bs)[-1]

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
        self.inference_mode = inference_mode

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
        fname_no_extension = op.splitext(base_fname)[0]
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
            Choose a random starting clothing segmentation from a different frame_num
            (.npy), and perform data augmention on each channel of that frame_num

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
        body_seg_file = (
            self._get_matching_body_seg_file(target_cs_file)
            if not self.inference_mode
            else self.body_seg_files[index]
        )
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
            body_s = crop(body_s, self.crop_bounds)
            input_cs = crop(input_cs, self.crop_bounds)
            target_cs = crop(target_cs, self.crop_bounds)

        return body_s, input_cs, target_cs


# TODO: lot of duplicated code. have to optimize this
class TextureDataset(Dataset):
    def __init__(
        self, texture_dir, rois_db, clothing_dir, min_offset=100, crop_bounds=None
    ):
        """
        Strategy:
            Get a target photo (.png). Get the matching clothing
            segmentation (.npy).


        HAVE TO UPDATE THE ROI TO MATCH CROP BOUNDS AND POSSIBLE NONE TYPES
        WHAT DO IF NONE? PUT ZERO?


        From the target image, get the matching clothing img.
        apply the SAME transform to the clothing img and target texture
        """
        super().__init__()

        self.texture_dir = texture_dir
        self.texture_files = os.listdir(texture_dir)

        rois_df = pd.read_csv(rois_db, index_col=False)
        # remove None values
        rois_df.replace("None", 0, inplace=True)
        rois_df = rois_df.astype(np.float16)
        rois_np = rois_df.values
        crop_rois(rois_np, crop_bounds)
        self.rois = torch.from_numpy(rois_np)
        self.clothing_dir = clothing_dir

        self.min_offset = min_offset
        self.crop_bounds = crop_bounds

    def _get_random_texture(self, index):
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
        if max_thresh >= len(self.texture_files):
            max_thresh = len(self.texture_files) - 1

        # our valid set
        valid_choices = (
            self.texture_files[:min_thresh] + self.texture_files[max_thresh:]
        )

        return random.choice(valid_choices)

    def get_matching_file(self, fname, dir, ext):
        """
        Given a filename, get the matching file in a different directory that has a
        different extension.
        :param fname:
        :param dir: other dir to get the file from
        :param ext:
        :return:
        """
        fname = op.basename(fname)
        fname_no_ext = op.splitext(fname)[0]
        cloth_name = fname_no_ext + ext
        return op.join(dir, cloth_name)

    def __len__(self):
        return len(self.texture_files)

    def get_matching_rois(self, index):
        """
        get matching roi based on index
        :return:
        """
        rois = self.rois[self.rois["id"] == index].values
        return rois

    def __getitem__(self, index):
        """
        Gets a clothing file and its corresponding texture file.
        :param index:
        :return:
        """
        texture_file = op.join(self.texture_dir, self.texture_files[index])
        texture_img = Image.open(texture_file)

        target_tex_file = op.join(self.texture_dir, self._get_random_texture(index))
        target_tex_img = Image.open(target_tex_file)

        cloth_file = self.get_matching_file(target_tex_file, self.clothing_dir, ".png")
        cloth_img = Image.open(cloth_file)

        if random.random() > 0.5:
            target_tex_img = t_func.hflip(target_tex_img)
            cloth_img = t_func.hflip(cloth_img)

        texture = to_tensor(texture_img)
        rois = self.get_matching_rois(index)
        cloth = to_tensor(cloth_img)
        target = to_tensor(target_tex_img)

        if self.crop_bounds:
            texture = crop(texture, self.crop_bounds)
            # ROI is already cropped
            cloth = crop(cloth, self.crop_bounds)
            target = crop(target, self.crop_bounds)

        return texture, rois, cloth, target
