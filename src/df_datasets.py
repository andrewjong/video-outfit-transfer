import os
import torch
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
import torchvision.transforms.functional as TF


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


# this parameter config is not from the paper.
swapnet_random_transform = random_transform_functional(transforms.RandomAffine(degrees=20, translate=(0.4, 0.4), scale=(0.75, 1.25), shear=10),
                                            transforms.RandomHorizontalFlip(0.3),
                                            transforms.RandomVerticalFlip(0.3),
                                           )


def to_onehot_tensor(sp_matrix, n_labels):
    """
    convert sparse scipy labels matrix to onehot pt tensor of size (n_labels,H,W)
    Note: sparse tensors aren't supported in multiprocessing https://github.com/pytorch/pytorch/issues/20248
    
    :param sp_matrix: sparse 2d scipy matrix, with entries in range(n_labels)
    :return: pt tensor of size(n_labels,H,W)
    """
    sp_matrix = sp_matrix.tocoo()
    indices = np.vstack((sp_matrix.data, sp_matrix.row, sp_matrix.col))
    indices = torch.LongTensor(indices)
    values = torch.Tensor([1.0]*sp_matrix.nnz)
    shape = (n_labels,) + sp_matrix.shape
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape)).to_dense()
    
class WarpDataset(Dataset):
    def __init__(
        self,
        body_seg_dir: str,
        cloth_seg_dir: str,
        crop_bounds: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        random_seed=None,
        input_transform=None,
        body_means=None,
        body_stds=None,
        body_ext: str='.jpg',
        cloth_ext: str='.npz',
        inference_mode=False,
    ):
        """
        Warp dataset for the warping module of SwapNet. All files in the dataset must be
        from the same subject (and ideally with the same background)

        :param cloth_seg_dir: path to directory containing cloth segmentation (.npz) files
        :param body_seg_dir: path to directory containing body segmentation image (.jpg) files
        :param input_transform: torchvision transform for the random drawn cloth segmentation image.
        
        """
        if random_seed:
            random.seed(random_seed)
        self.random_seed = random_seed
        
        self.cloth_ext = cloth_ext
        self.body_ext = body_ext
        
        self.cloth_seg_dir = cloth_seg_dir
        os.chdir(self.cloth_seg_dir)
        self.cloth_seg_files = glob(('**/*'+self.cloth_ext), recursive=True)
        os.chdir('../'*(len(self.cloth_seg_dir.split('/'))))
        
        self.body_seg_dir = body_seg_dir

        self.crop_bounds = crop_bounds

        self.input_transform = input_transform
        
        # transforms for RGB images
        body_transform = [transforms.ToTensor()]
        if body_means and body_stds:
            body_transform += [transforms.Normalize(body_means, body_stds)]
        self.body_transform = transforms.Compose(body_transform)
            
        self.inference_mode = inference_mode
        
    def _change_extension(self, fname, ext1, ext2):
        """
        :return: file name with new extension
        """
        return fname[:-len(ext1)] + ext2
    
    
    def _decompress_cloth_segment(self, fname, n_labels) -> Tensor:
        """
        load cloth segmentation sparse matrix npz file
        :return: tensor of size(H,W,n_labels)
        """
        data_sparse = load_npz(fname)
        return to_onehot_tensor(data_sparse, n_labels)
        
    
    def _perchannel_transform(self, input_cloth_np, transform_function) -> Tensor:
        """
        Randomly transform each of n_channels of input data.
        Out of place operation

        :param input_cloth_np: must be a numpy array of size (n_channels, w, h)
        :param transform_function: any torchvision transforms class
        :return: transformed pt tensor
        """
        tform_input_cloth_np = np.zeros(shape=input_cloth_np.shape, dtype=input_cloth_np.dtype)
        n_channels = input_cloth_np.shape[0]
        for i in range(n_channels):
            tform_input_cloth_np[i] = np.array(transform_function(Image.fromarray(input_cloth_np[i])))
        return torch.from_numpy(tform_input_cloth_np)
        

    def __len__(self):
        """
        Get the length of usable images. Note the length of cloth and body segmentations should be same
        :return: length of the image
        """
        return len(self.cloth_seg_files)


    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Strategy:
            Get a target cloth segmentation (.npz). Get the matching body
            segmentation (.jpg)
            Choose a random input cloth segmentation from a different frame_num
            (.npz), and perform data augmention on each channel of that frame_num

        :returns:
            For training, return (input) AUGMENTED cloth seg, (input) body seg and (output) cloth seg
            of the SAME person
            For inference (e.g validation), return (input) cloth seg and (input) body seg
            of 2 different people
        """

        # the target cloth segmentation
        target_cloth_file = self.cloth_seg_files[index]
        target_cloth_tensor = self._decompress_cloth_segment(os.path.join(self.cloth_seg_dir, target_cloth_file), n_labels=19)

        # the input cloth segmentation
        input_cloth_file = random.choice(self.cloth_seg_files)
        while input_cloth_file == target_cloth_file:
            input_cloth_file = random.choice(self.cloth_seg_files) # prevent getting the same pose
            
        input_cloth_tensor = self._decompress_cloth_segment(os.path.join(self.cloth_seg_dir, input_cloth_file), n_labels=19)

        # the body segmentation that corresponds to the target cloth segmentation
        if not self.inference_mode:
            # get the same target image's body segmentation
            input_body_file = self._change_extension(target_cloth_file, self.cloth_ext, self.body_ext)
        else:
            input_body_file = random.choice(self.cloth_seg_files)
            input_body_file = self._change_extension(input_body_file, self.cloth_ext, self.body_ext)
    
        input_body_img = Image.open(os.path.join(self.body_seg_dir, input_body_file))
        input_body_tensor = self.body_transform(input_body_img)
        
#         print(input_body_file, input_cloth_file, target_cloth_file)

        # apply the transformations if desired
        if self.input_transform and self.inference_mode:
            input_cloth_np = input_cloth_tensor.numpy()
            input_cloth_tensor = self._perchannel_transform(input_cloth_np, self.input_transform)


        # crop to the proper image size
        if self.crop_bounds:
            input_body_tensor = crop(input_body_tensor, self.crop_bounds)
            input_cloth_tensor = crop(input_cloth_tensor, self.crop_bounds)
            target_cloth_tensor = crop(target_cloth_tensor, self.crop_bounds)

        if not self.inference_mode:
            return input_body_tensor, input_cloth_tensor, target_cloth_tensor
        else:
            return input_body_tensor, input_cloth_tensor,

    
    
class TextureDataset(Dataset):
    def __init__(self, 
                 texture_dir: str,
                 rois_db: str,
                 cloth_seg_dir: str,
                 random_seed: int = None,
                 input_transform =None, # should default to swapnet transform?
                 crop_bounds = None,
                 cloth_ext: str = '.npz',
                 img_ext: str = '.jpg',
                 inference_mode=False,
                ):
        if random_seed:
            random.seed(random_seed)
        super().__init__()
        self.texture_dir = texture_dir
        self.cloth_seg_dir = cloth_seg_dir
        self.img_ext = img_ext
        self.cloth_ext = cloth_ext
        
        os.chdir(self.texture_dir)
        self.texture_files = glob(os.path.join('**/*'+self.img_ext), recursive=True)
        os.chdir('../'*(len(self.texture_dir.split('/'))))
        
        self.rois_df = pd.read_csv(rois_db, index_col=0)
        
        self.rois_df = self.rois_df.replace("None", 0).astype(np.float32)
        
        self.input_transform = input_transform
        self.crop_bounds = crop_bounds
        self.inference_mode = inference_mode
        
        
    def _decompress_cloth_segment(self, fname, n_labels) -> torch.Tensor:
        """
        load cloth segmentation from sparse matrix npz file
        :return: sparse tensor of size(H,W,n_labels)
        """
        data_sparse = load_npz(fname)
        
        return to_onehot_tensor(data_sparse, n_labels)
    
    
    # TODO: complete this
#     def _random_crop_and_flip(img, rois=None):
#         H, W = img.shape[1], img.shape[2]
#         if random.random() < 0.3:
#             TF.hflip(img)
#             if rois:
                
            
#         if random.random() < 0.3:
#             TF.vflip(img)
#             if rois:
        
        
    def __len__(self):
        return len(self.texture_files)
    
    def __getitem__(self, index: int):
        """
        Q: are we missing the target face and other details? 
        A: Recall that we only warp cloth, everything else from the texture is copied over (post-process)
        
        Returns:
            (augmented) input texture with desired cloth
            rois of (augmented) input texture
            cloth segmentation at the desired pose
            input texture with desired cloth (for training)
        """
        input_texture_file = self.texture_files[index]
        input_texture_img = Image.open(os.path.join(self.texture_dir, input_texture_file))
        
        # replace .png with .npz
        file_name = input_texture_file[:-len(self.img_ext)]
        input_cloth_file = file_name + self.cloth_ext
        input_cloth_tensor = self._decompress_cloth_segment(os.path.join(self.cloth_seg_dir, input_cloth_file), n_labels=19)
        
        # refer to swapnet p.9
        if self.input_transform and self.inference_mode:
            input_texture_tensor = to_tensor(self.input_transform(input_texture_img))
        else:
            input_texture_tensor = to_tensor(input_texture_img)
        
        output_texture_tensor = to_tensor(input_texture_img)
            
        #TODO: We should remove None rois preemptively
        # otherwise I can only think if awkward way to handle it inside __getitem__
        # for now I'm passing 0
        rois = self.rois_df.loc[file_name].values
        input_rois_tensor = torch.from_numpy(rois)
        
        if self.crop_bounds:
            input_texture_tensor = crop(input_texture_tensor, self.crop_bounds)
            input_rois_tensor = crop_rois(input_rois_tensor, self.crop_bounds)
            input_cloth_tensor = crop(input_cloth_tensor, self.crop_bounds)
        
        if not self.inference_mode:
            return input_texture_tensor, input_rois_tensor, input_cloth_tensor, output_texture_tensor
        else:
            return input_texture_tensor, input_rois_tensor, input_cloth_tensor,
        