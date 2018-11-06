import torch
import torch.nn as nn



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



