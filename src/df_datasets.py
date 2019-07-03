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
    def transform(input_cloth_np) -> Tensor:
        """
        Randomly transform each of n_channels of input data.
        Out of place operation

        :param: input_cloth_np: must be a PIL Image of size (n_channels, w, h)
        :return: new copy of transformed PIL Image
        """
        print(input_cloth_np.shape)
        tform_input_cloth_np = np.zeros(shape=input_cloth_np.shape, dtype=input_cloth_np.dtype)
        n_channels = input_cloth_np.shape[2]
        for i in range(n_channels):
            tform_input_cloth_np[:,:,i] = np.array(transform_function(Image.fromarray(input_cloth_np[:,:,i])))
        return torch.from_numpy(tform_input_cloth_np)
    return transform

# this parameter config is not from the paper.
swapnet_random_transform = random_transform_functional(transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), shear=30),
                                            transforms.RandomHorizontalFlip(0.3),
                                            transforms.RandomVerticalFlip(0.3),
                                           )

swapnet_per_channel_transform = random_per_channel_transform_functional(swapnet_random_transform)
    

# def to_onehot(sp_matrix, n_labels):
#     """
#     convert ndarray labels to dense onehot ndarray
#     last dimension must be in range(n_labels)
#     """
#     oshape = img.shape
#     img = img.reshape(np.prod(img.shape), 1)
#     res = np.zeros(shape=(sp_matrix.shape[0][:,None], n_labels))
#     res[np.arange(img.shape[0]).T,img] = 1.0
#     res = res.reshape(oshape+(n_labels,))
#     return res
    

def to_onehot_sparse_tensor(sp_matrix, n_labels):
    """
    convert sparse scipy labels matrix to sparse onehot pt tensor
    last dimension must be in range(n_labels)
    """
    sp_matrix = sp_matrix.tocoo()
    indices = np.vstack((sp_matrix.row, sp_matrix.col, sp_matrix.data))
    indices = torch.LongTensor(indices)
    values = torch.Tensor([1.0]*sp_matrix.nnz)
    shape = sp_matrix.shape + (n_labels,)
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    
class WarpDataset(Dataset):
    def __init__(
        self,
        body_seg_dir: str,
        cloth_seg_dir: str,
        crop_bounds: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        random_seed=None,
        input_transform=swapnet_per_channel_transform,
        body_means=None,
        body_stds=None,
        inference_mode=False,
        body_ext: str='.png',
        cloth_ext: str='.npz',
    ):
        """
        Warp dataset for the warping module of SwapNet. All files in the dataset must be
        from the same subject (and ideally with the same background)

        :param cloth_seg_dir: path to directory containing cloth segmentation
        .npy files
        :param body_seg_dir: path to directory containing body segmentation image files
        :param min_offset: minimum offset to select a random other cloth
        segmentation. (default: 50; unless min_offset < number of cloth files,
        then 0)
        :param random_seed: seed for getting a random cloth seg image
        :param input_transform: transform for the random drawn cloth segmentation image.
        Note, the transform must be able to operate on a HxWx19-channel tensor
        """
        if random_seed:
            random.seed(random_seed)
        self.random_seed = random_seed
        
        self.cloth_ext = cloth_ext
        self.body_ext = body_ext
        
        # TODO: cleaner way to get rid of prefix dir
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
    
    
    def _decompress_cloth_segment(self, fname, n_labels) -> torch.sparse.FloatTensor:
        """
        load cloth segmentation sparse matrix npz file
        """
        data_sparse = load_npz(fname)
        
        return to_onehot_sparse_tensor(data_sparse, n_labels)
    
        

    def __len__(self):
        """
        Get the length of usable images. Note the length of cloth and body segmentations should be same
        :return: length of the image
        """
        return len(self.cloth_seg_files)


    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Strategy:
            Get a target cloth segmentation (.npy). Get the matching body
            segmentation (.png)
            Choose a random starting cloth segmentation from a different frame_num
            (.npy), and perform data augmention on each channel of that frame_num

        :param index:
        :return: input body segmentation, input cloth segmentation, target cloth
        segmentation
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
        
        print(input_body_file, input_cloth_file, target_cloth_file)

        # apply the transformations if desired
        if self.input_transform:
            input_cloth_np = input_cloth_tensor.to_dense().numpy()
            input_cloth_tensor = self.input_transform(input_cloth_np)


        # crop to the proper image size
        if self.crop_bounds:
            input_body_tensor = crop(input_body_tensor, self.crop_bounds)
            input_cloth_tensor = crop(input_cloth_tensor, self.crop_bounds)
            target_cloth_tensor = crop(target_cloth_tensor, self.crop_bounds)

        return input_body_tensor, input_cloth_tensor, target_cloth_tensor

    
    
class TextureDataset(Dataset):
    def __init__(self, 
                 texture_dir: str,
                 rois_db: str,
                 cloth_seg_dir: str,
                 random_seed: int = None,
                 input_transform = swapnet_random_transform, # should default to swapnet transform?
                 crop_bounds = None,
                 cloth_ext: str = '.npz',
                 img_ext: str = '.jpg',
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
        
        self.rois_df = pd.read_csv(rois_db, index_col=False)
        self.rois_df = self.rois_df.replace("None", 0).astype(np.float32)
        
        self.input_transform = input_transform
        self.crop_bounds = crop_bounds
        
        
    def _decompress_cloth_segment(self, fname, n_labels) -> torch.sparse.FloatTensor:
        """
        load cloth segmentation sparse matrix npz file
        """
        data_sparse = load_npz(fname)
        
        return to_onehot_sparse_tensor(data_sparse, n_labels)
        
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
        if self.input_transform:
            input_texture_tensor = to_tensor(self.input_transform(input_texture_img))
        else:
            input_texture_tensor = to_tensor(input_texture_img)
        
        output_texture_tensor = to_tensor(input_texture_img)
            
        rois = self.rois_df[self.rois_df['id'] == file_name].values
        input_rois_tensor = torch.from_numpy(rois)
        
        if self.crop_bounds:
            input_texture_tensor = crop(input_texture_tensor, self.crop_bounds)
            input_rois_tensor = crop_rois(input_rois_tensor, self.crop_bounds)
            input_cloth_tensor = crop(input_cloth_tensor, self.crop_bounds)
        
        return input_texture_tensor, input_rois_tensor, input_cloth_tensor, output_texture_tensor
        