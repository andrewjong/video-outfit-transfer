import sys


sys.path.append("/home/remoteuser/faster-rcnn.pytorch/lib")
import torch
from model.roi_layers import ROIAlign
from src.datasets import TextureDataset


roi_align = ROIAlign((10, 10), spatial_scale=1 / 16, sampling_ratio=0)
texture_dataset = TextureDataset(
    texture_dir="data/andrew/texture",
    rois="data/andrew/rois.csv",
    clothing_dir="data/andrew/clothing",
)


dataloader = torch.utils.data.DataLoader(texture_dataset, batch_size=2)

first_batch = next(iter(dataloader))

tex_img_and_roi = first_batch[:2]


