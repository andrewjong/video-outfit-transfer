# I think we'd do these two steps offline to generate our dataset
# Apply this on video
# TODO import LIP_SSL for clothing segmentation
# TODO import Unite the People for body segmentation


from .nets import *


class ClothingEncoder(nn.Module):
    def __init__(self, channels=18):
        super(ClothingEncoder, self).__init__()

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)

    def forward(self, x):
        # Propogate noise through fc layer and reshape to img shape
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        # # u4 = self.up4(u3, d3)
        # # u5 = self.up5(u4, d2)
        # # u6 = self.up6(u5, d1)
        #
        # return self.final(u6)
        return u3


class PoseEncoder(nn.Module):
    def __init__(self, channels=3):
        super(PoseEncoder, self).__init__()

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)

    def forward(self, x):
        # Propogate noise through fc layer and reshape to img shape
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        return d6


class WarpingModule(nn.Module):
    """
    Clothing encoder, Pose encoder

    Arguments:
        nn {[type]} -- [description]
    ""[summary]
    """

    pass

    def __init__(self):
        super(WarpingModule, self).__init__()
        self.pose_encoder = PoseEncoder()
        self.clothing_encoder = ClothingEncoder()
        self.resblocks = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        # note, a traditional U-Net would use the UNetUp blocks. But UNetUp blocks
        # require concatenation via skip connection of a previous UNetDown layer.
        # Since this is a dual path U-Net, I'm not sure which path should be included
        #  in the skip. Therefore I opt for simply transpose convolutions
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 18, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, target_human, target_clothing):
        encoded_pose = self.pose_encoder(target_human)
        encoded_clothing = self.clothing_encoder(target_clothing)
        encoded = torch.cat((encoded_pose, encoded_clothing))

        # 4 residual blocks
        x = self.resblocks(encoded)

        # Upsample
        x = self.upsample(x)
        return x


