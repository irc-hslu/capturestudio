import torch
from torch import nn

from reconstruction.arch.gpsgaussian.extractor import UnetExtractor, ResidualBlock


# noinspection PyTypeChecker
class GSRegresser(nn.Module):
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group', scale_tanh: bool = False, scale_norm: bool = False):
        super().__init__()
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2] + self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1] + self.depth_dims[1] + self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0] + self.depth_dims[0] + self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0] + rgb_dim + depth_dim, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        scale_head_modules = [
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
        ]
        self.scale_clamp_min = 0.0
        self.scale_clamp_max = 0.01
        if scale_tanh or scale_norm:
            scale_head_modules.append(nn.Tanh())
            self.scale_clamp_min = 0.0001
        if scale_norm:
            scale_head_modules.append(nn.InstanceNorm2d(3, affine=True))
            self.scale_clamp_min = 0.0001
        if not scale_tanh and not scale_norm:
            scale_head_modules.append(nn.Softplus(beta=100))
        self.scale_head = nn.Sequential(*scale_head_modules)
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img, depth, img_feat, alpha=1.0, beta=0.0):
        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)

        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = (alpha * self.scale_head(out) + beta).clamp(self.scale_clamp_min, self.scale_clamp_max)# * 0.01 + 0.0025

        # opacity head
        opacity_out = self.opacity_head(out)

        return rot_out, scale_out, opacity_out
