import os
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import copy
import torchvision
from glob import glob
import torchvision.transforms as transforms
from scipy.sparse import load_npz
import pandas as pd


to_tensor = torchvision.transforms.ToTensor()
# can we reuse WarpDataset?

def random_transform(*args):
    return transforms.Compose(args)

# this parameter config is not from the paper.
swapnet_random_transform = random_transform(transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)),
                                            transforms.RandomHorizontalFlip(0.3),
                                            transforms.RandomVerticalFlip(0.3),
                                            transforms.ToTensor(),
                                           )

class TextureDataset(Dataset):
    def __init__(self, 
                 image_root: str,
                 rois_db: str,
                 cloth_seg_root: str,
                 random_seed: int = None,
                 input_transform = None, # should default to swapnet transform?
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
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index: int):
        img_input_path = self.image_files[index]
        
        # exclude data-specific path name
        offset = len(self.image_root) + 1
        common_path = img_input_path[offset:-4]
        # replace .png with .npz
        cloth_input_path = common_path + self.ext
        cloth_input_path = os.path.join(self.cloth_seg_root, cloth_input_path)
        
        img_input_data = Image.open(img_input_path)
        cloth_cs_input_data = load_npz(cloth_input_path).todense()
        cloth_cs_input_data = Image.fromarray(cloth_cs_input_data) # how to avoid this?
        
        if self.input_transform:
            img_input_tensor = self.input_transform(img_input_data)
            cloth_cs_input_tensor = self.input_transform(cloth_cs_input_data)
        else:
            img_input_tensor = to_tensor(img_input_data)
            cloth_cs_input_tensor = to_tensor(cloth_cs_input_data)
            
        rois = self.rois_df[self.rois_df['id'] == common_path].values
        rois_input_tensor = torch.from_numpy(rois)
        
        return img_input_tensor, rois_input_tensor, cloth_cs_input_tensor
        