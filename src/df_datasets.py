import os
from torch.utils.data import Dataset
from torch import Tensor
import random
from PIL import Image
import numpy as np
import copy
import torchvision
from glob import glob
import torchvision.transforms as transforms
from scipy.sparse import load_npz
import pandas as pd
from typing import Set, List, Tuple


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

to_tensor = torchvision.transforms.ToTensor()


def random_transform_functional(*args):
    return transforms.Compose(args)


def random_per_channel_transform_functional(transform_function):
    def transform_function(input_cloth_img):
        """
        Randomly transform each of n_channels of input data.
        Out of place operation

        :param: input_cloth_img: must be a PIL Image of size (n_channels, w, h)
        :return: new copy of transformed PIL Image
        """
        input_cloth_data = input_cloth_img.getdata()
        tform_input_cloth_data = np.zeros(shape=input_cloth_data.shape, dtype=input_cloth_data.dtype)
        n_channels = len(input_cloth_data)
        for i in range(n_channels):
            tform_input_cloth_data[i] = transform_function(Image.fromarray(input_cloth_data[i]))
        return Image.fromarray(tform_input_cloth_data)
    return transform_function

# this parameter config is not from the paper.
swapnet_random_transform = random_transform_functional(transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), shear=30),
                                            transforms.RandomHorizontalFlip(0.3),
                                            transforms.RandomVerticalFlip(0.3),
                                           )

swapnet_per_channel_transform = random_per_channel_transform_functional(swapnet_random_transform)
    

class WarpDataset(Dataset):
    def __init__(
        self,
        body_seg_dir: str,
        clothing_seg_dir: str,
        crop_bounds: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        random_seed=None,
        input_transform=None,
        body_means=None,
        body_stds=None,
        inference_mode=False,
        body_ext: str='.png',
        cloth_ext: str='.npz',
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
        if random_seed:
            random.seed(random_seed)
        self.random_seed = random_seed
        
        self.cloth_ext = cloth_ext
        self.body_ext = body_ext
        
        # TODO: cleaner way to get rid of prefix dir
        self.clothing_seg_dir = clothing_seg_dir
        os.chdir(self.clothing_seg_dir)
        self.clothing_seg_files = glob(('**/*'+self.cloth_ext), recursive=True)
        os.chdir('../'*(len(self.clothing_seg_dir.split('/'))))
        
        self.body_seg_dir = body_seg_dir

        self.crop_bounds = crop_bounds

        self.input_transform = input_transform
        
        # transforms for RGB images
        body_transforms = [torchvision.transforms.ToTensor()]
        if body_means and body_stds:
            body_transforms.append(
                torchvision.transforms.Normalize(body_means, body_stds)
            )
        self.body_transforms = torchvision.transforms.Compose(body_transforms)
        self.inference_mode = inference_mode
        
    def _change_extension(self, fname, ext1, ext2):
        """
        Return:
            file name with new extension
        """
        return fname[:-len(ext1)] + ext2
    
    def _load_by_ext(self, fname, ext, type='PIL'):
        """
        Choose load method according to extension
        
        :param ext: input file extension
        :param dtype: output file data type
        :return: loaded data a
        """
        
        if ext in ['.jpg', '.png']:
            img = Image.open(fname)
        elif ext == '.npz':
            img_np = load_npz(fname).todense()
            img = Image.fromarray(img_np)
        return img
        

    def __len__(self):
        """
        Get the length of usable images. Note the length of clothing and body segmentations should be same
        :return: length of the image
        """
        return len(self.clothing_seg_files)


        return cloth_input_path
    
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

        # the target clothing segmentation
        target_cloth_file = self.clothing_seg_files[index]
        target_cloth_img = self._load_by_ext(os.path.join(self.clothing_seg_dir, target_cloth_file), self.cloth_ext)

        # the input clothing segmentation
        input_cloth_file = random.choice(self.clothing_seg_files)
        while input_cloth_file == target_cloth_file:
            input_cloth_file = random.choice(self.clothing_seg_files) # prevent getting the same pose
            
        input_cloth_img = self._load_by_ext(os.path.join(self.clothing_seg_dir, input_cloth_file), self.cloth_ext)

        # the body segmentation that corresponds to the target clothing segmentation
        if not self.inference_mode:
            # get the same target image's body segmentation
            input_body_file = self._change_extension(target_cloth_file, self.cloth_ext, self.body_ext)
        else:
            input_body_file = random.choice(self.clothing_seg_files)
#             while input_body_file == input_cloth_file:
#                 input_body_file = random.choice(selfe.clothing_seg_files)
            input_body_file = self._change_extension(input_body_file, self.cloth_ext, self.body_ext)
    
        input_body_img = self._load_by_ext(os.path.join(self.body_seg_dir, body_seg_file), self.body_ext)
        
        # print(input_cloth_file, input_body_file, target_cloth_file)

        # apply the transformations if desired
        if self.input_transform:
            input_cloth_img = self.input_transform(input_cloth_img)

        # convert to tensors
        input_body_tensor = self.body_transforms(input_body_img)
        input_cloth_tensor = to_tensor(input_cloth_img)
        target_cloth_tensor = to_tensor(target_cloth_img)

        # crop to the proper image size
        if self.crop_bounds:
            input_body_tensor = crop(input_body_tensor, self.crop_bounds)
            input_cloth_tensor = crop(input_cloth_tensor, self.crop_bounds)
            target_cloth_tensor = crop(target_cloth_tensor, self.crop_bounds)

        return input_body_tensor, input_cloth_tensor, target_cloth_tensor

    
    
class TextureDataset(Dataset):
    def __init__(self, 
                 image_root: str,
                 rois_db: str,
                 cloth_seg_root: str,
                 random_seed: int = None,
                 input_transform = None, # should default to swapnet transform?
                 crop_bounds = None,
                 ext: str = '.npz'
                ):
        if random_seed:
            random.seed(random_seed)
        super().__init__()
        self.image_root = image_root
        self.image_files = glob(os.path.join(self.image_root, '**/*.jpg'), recursive=True)
        self.rois_df = pd.read_csv(rois_db, index_col=False)
        self.rois_df = rois_df.replace("None", 0).astype(np.float32)
        self.cloth_seg_root = cloth_seg_root
        self.input_transform = input_transform
        self.ext = ext
        # self.cloth_seg_files =set(glob(self.cloth_seg_root+'/**/*.npz', recursive=True))
        
    def _load_by_ext(self, fname, ext):
        """
        Choose load method according to extension 
        
        Return:
            loaded data as PIL Image
        """
        if ext in ['.jpg', '.png']:
            img = Image.open(fname)
        elif ext == '.npz':
            img_np = load_npz(fname).todense()
            img = Image.fromarray(img_np)
        return img
        
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index: int):
        """
        are we missing the target face and other details? 
        Recall that we only warp cloth, everything else from the image is copied over
        
        Returns:
            (augmented) input image with desired clothing
            rois of (augmented) input image
            cloth segmentation at the desired pose
            input image with desired clothing (for training)
        """
        img_input_path = self.image_files[index]
        
        # exclude data-specific path name
        offset = len(self.image_root) + 1
        common_path = img_input_path[offset:-4]
        # replace .png with .npz
        cloth_input_path = common_path + self.ext
        cloth_input_path = os.path.join(self.cloth_seg_root, cloth_input_path)
        
        img_input_data = Image.open(img_input_path)
        cloth_input_data = load_npz(cloth_input_path).todense()
        cloth_input_data = Image.fromarray(cloth_input_data) # how to avoid this?
        
        # refer to swapnet p.9
        if self.input_transform:
            img_input_tensor = to_tensor(self.input_transform(img_input_data))
        else:
            img_input_tensor = to_tensor(img_input_data)
        
        cloth_input_tensor = to_tensor(cloth_cs_input_data)
        img_output_tensor = to_tensor(img_input_data)
            
        rois = self.rois_df[self.rois_df['id'] == common_path].values
        rois_input_tensor = torch.from_numpy(rois)
        
        if self.crop_bounds:
            texture = crop(texture, self.crop_bounds)
            rois = crop_rois(rois, self.crop_bounds)
            cloth = crop(cloth, self.crop_bounds)
        
        return img_input_tensor, rois_input_tensor, cloth_input_tensor, img_output_tensor
        