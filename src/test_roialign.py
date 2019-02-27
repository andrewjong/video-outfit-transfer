import sys


sys.path.append("/home/remoteuser/faster-rcnn.pytorch/lib")
import torch
from model.roi_layers import ROIAlign
from src.datasets import TextureDataset


roi_align = ROIAlign((10, 10), spatial_scale=1 / 16, sampling_ratio=0)
texture_dataset = TextureDataset(
    texture_dir="data/andrew/texture",
    roi_dir="data/andrew/rois",
    clothing_dir="data/andrew/clothing",
)
first = texture_dataset[0]

# cuda = True if torch.cuda.is_available() else False
# if cuda:
#     torch.cuda.set_device(0)

img = first[0]
expanded = img.expand(1, -1, -1, -1)
my_roi = first[1]

# THIS DOESN'T WORK, WRONG DTYPE
# roi_thing = torch.tensor([[0, 0, 0, 50, 50], [0, 0, 0, 50, 50]])
# THIS DOES, explicitly set dtype

roi_thing = torch.tensor([[0, 0, 0, 50, 50], [0, 0, 0, 50, 50]], dtype=torch.float32)
roi_align(expanded, roi_thing)


roi_align = ROIAlign((10, 10), spatial_scale=1 / 16, sampling_ratio=0)
input = torch.zeros([1, 1024, 38, 56])
rois = torch.tensor([[0, 329.27, 0, 891.375, 599.0], [0.0, 151.45, 87.65, 895, 484.26]])
# input = input.cuda()
# rois = rois.cuda()



# learned that cuda shouldn't matter

# learned that dtype must be float.32, not int.64. If int.64, will cause
# THPFunction_apply  apply error or something

# learned that MY IMAGE with THEIR ROI frickin works. Means something must be wrong
# with my roi. Have to isolate specifically what.


# FOUND WHAT CAUSES BREAKAGE
# if all img_id == 0, then it works.
# else if any other number, then it doesn't work

# hmmmm, this makes sense if it's related to the image number in the BATCH SIZE. we
# were using batch size 1. So of course we can't access anything other than imgid [0]



dataloader = torch.utils.data.DataLoader(
    texture_dataset, batch_size=3, num_workers=1
)

first_batch = next(iter(dataloader))
first_batch[1][0, :, :][:,0]=0 # set to 0
first_batch[1][1, :, :][:,0]=1 # set to 1
first_batch[1][2, :, :][:,0]=2 # set to 2
rois = first_batch[1].view(-1, 5)
imgs = first_batch[0]

roi_align(imgs, rois) # works!!!
# CONFIRMED!!!! THE FIRST INDEX IS BATCH INDEX, HUZZAH
# this means roi shouldn't be 1-6, but rather the id of the image


# Also need the compile-fastercnn-pytorch environment activated. probably need to
# install the correct dependencies
