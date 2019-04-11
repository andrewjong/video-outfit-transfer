import torch
import sys

sys.path.append("../lib")
from model.roi_layers import ROIAlign
from torch import nn

from src.nets import UNetDown, UNetUp

NUM_ROI = 6


class TextureModule(nn.Module):
    def __init__(self, texture_channels=3, dropout=0.5):
        super(TextureModule, self).__init__()
        self.roi_align = ROIAlign(
            output_size=(128, 128), spatial_scale=1, sampling_ratio=1
        )

        channels = texture_channels * NUM_ROI
        self.encode = UNetDown(channels, channels)

        # UNET
        self.down_1 = UNetDown(channels + texture_channels, 64, normalize=False)
        self.down_2 = UNetDown(64, 128)
        self.down_3 = UNetDown(128, 256)
        self.down_4 = UNetDown(256, 512, dropout=dropout)
        self.down_5 = UNetDown(512, 1024, dropout=dropout)
        self.down_6 = UNetDown(1024, 1024, normalize=False, dropout=dropout)
        self.up_1 = UNetUp(1024, 1024, dropout=dropout)
        self.up_2 = UNetUp(2 * 1024, 512, dropout=dropout)
        self.up_3 = UNetUp(2 * 512, 256)
        self.up_4 = UNetUp(2 * 256, 128)
        self.up_5 = UNetUp(2 * 128, 64)

        self.upsample_and_pad = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, texture_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, input_tex, rois, cloth):
        # view rois as -1 5 because they have an extra "batch" dimension. but
        # roi_align api just expects a single rowsxcoords tensor
        rois = rois.view(-1, 5)
        rois[:, 0] = rois[:, 0] - rois[0, 0]
        pooled_rois = self.roi_align(input_tex, rois)
        # reshape the pooled rois such that pool output goes in the channels instead of
        # batch size
        batch_size = int(pooled_rois.shape[0] / NUM_ROI)
        pooled_rois = pooled_rois.view(
            batch_size, -1, pooled_rois.shape[2], pooled_rois.shape[3]
        )

        encoded_tex = self.encode(pooled_rois)

        scale_factor = input_tex.shape[2] / encoded_tex.shape[2]
        upsampled_tex = nn.functional.interpolate(
            encoded_tex, scale_factor=scale_factor
        )

        # concat on the channel dimension
        tex_with_cloth = torch.cat((upsampled_tex, cloth), 1)
        d1 = self.down_1(tex_with_cloth)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)
        d5 = self.down_5(d4)
        d6 = self.down_6(d5)
        u1 = self.up_1(d6, d5)
        u2 = self.up_2(u1, d4)
        u3 = self.up_3(u2, d3)
        u4 = self.up_4(u3, d2)
        u5 = self.up_5(u4, d1)

        return self.upsample_and_pad(u5)
