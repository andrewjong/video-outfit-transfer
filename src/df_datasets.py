import os
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import copy
import torchvision
from glob import glob


to_tensor = torchvision.transforms.ToTensor()
# can we reuse WarpDataset?

class TextureDataset(Dataset):
    def __init__(self, 
                 image_root: str,
                 cloth_seg_root: str,
                 random_seed: int = None,
                 ext: str = '.npz'
                ):
        if random_seed:
            random.seed(random_seed)
        super().__init__()
        self.image_root = image_root
        self.image_files = glob(self.image_root+'/**/*.jpg', recursive=True)
        self.cloth_seg_root = cloth_seg_root
        self.ext = ext
        # self.cloth_seg_files =set(glob(self.cloth_seg_root+'/**/*.npz', recursive=True))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index: int):
        img_input_path = self.image_files[index]
        
        # exclude data-specific path name
        offset = len(self.image_root)+1
        common_path = img_input_path[offset:]
        print(common_path)
        # replace .png with .npz
        cloth_input_path = common_path[:-4] + self.ext
        cloth_input_path = os.path.join(self.cloth_seg_root, cloth_input_path)
        print(cloth_input_path)
        
        img_input_data = Image.open(img_input_path)
        cloth_cs_input_data = np.load(cloth_input_path)
        img_output_data = copy.copy(img_input_data)
        
        img_input_tensor = to_tensor(img_input_data)
        cloth_cs_input_tensor = to_tensor(cloth_cs_input_data)
        img_output_tensor = to_tensor(img_output_data)
        
        return img_input_tensor, cloth_cs_input_tensor, img_output_tensor
        