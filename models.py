import torch
import torch.nn as nn

# I think we'd do these two steps offline to generate our dataset
# Apply this on video
# TODO import LIP_SSL for clothing segmentation
# TODO import Unite the People for body segmentation



class ClothingEncoder(nn.Module):
    def __init__(self, img_w_h):
        pass

class PoseEncoder(nn.Module):
    def __init__(self, img_w_h):
        pass

class WarpingModule(nn.Module):
    """
    Clothing encoder, Pose encoder
    
    Arguments:
        nn {[type]} -- [description]
    ""[summary]
    """
    pass
    def __init__(self, img_size):
        in_feat = 18 * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(in_feat, )
        )
        self.clothing_encoder = ClothingEncoder(img_size)
        self.pose_encoder = PoseEncoder(img_size)

    def forward(self, target_human, target_clothing):
        encoded_pose = self.pose_encoder(target_human)
        encoded_clothing = self.clothing_encoder(target_clothing)
        encoded = torch.cat((encoded_pose, encoded_clothing))
        # This is the remaining part of the network
        x = self.something(encoded)
        x = self.something(encoded)
        x = self.upsample(encoded)
        return x


class TextureModule(nn.Module):
    """Does the texture part of it
    
    Arguments:
        nn {[type]} -- [description]
    """

class SwapNet(nn.Module):
    """[summary]
    
    Arguments:
        nn {[type]} -- [description]
    """



