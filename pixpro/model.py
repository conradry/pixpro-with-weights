import random, math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict

class PPM(nn.Module):
    def __init__(self, nin, gamma=2, nlayers=1):
        super(PPM, self).__init__()

        #the networks can have 0-2 layers
        if nlayers == 0:
            layers = [nn.Identity()]
        elif nlayers == 1:
            layers = [nn.Conv2d(nin, nin, 1)]
        elif nlayers == 2:
            layers = [
                nn.Conv2d(nin, nin, 1, bias=False),
                nn.BatchNorm2d(nin),
                nn.ReLU(),
                nn.Conv2d(nin, nin, 1)
            ]
        else:
            raise Exception(f'nlayers must be 0, 1, or 2, got {nlayers}')

        self.transform = nn.Sequential(*layers)
        self.gamma = gamma
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        #(B, C, H, W, 1, 1) x (B, C, 1, 1, H, W) --> (B, H, W, H, W)
        #relu is same as max(sim, 0) but differentiable
        xi = x[:, :, :, :, None, None]
        xj = x[:, :, None, None, :, :]
        s = F.relu(self.cosine_sim(xi, xj)) ** self.gamma

        #output of "g" in the paper (B, C, H, W)
        gx = self.transform(x)

        #use einsum and skip all the transposes
        #and matrix multiplies
        y = torch.einsum('bijhw, bchw -> bcij', s, gx)

        return x

class Encoder(nn.Module):
    def __init__(self, backbone):
        super(Encoder, self).__init__()
        
        #use ordered dict to retain layer names
        #last two layers are avgpool and fc
        layers = OrderedDict([layer for layer in backbone().named_children()][:-2])
        self.backbone = nn.Sequential(layers)
        
        #get the number of channels from last convolution
        for layer in self.backbone.modules():
            if isinstance(layer, nn.Conv2d):
                projection_nin = layer.out_channels

        #note that projection_nin = 2048 for resnet50 (matches paper)
        self.projection = nn.Sequential(
            nn.Conv2d(projection_nin, projection_nin, 1, bias=False),
            nn.BatchNorm2d(projection_nin),
            nn.ReLU(inplace=True),
            nn.Conv2d(projection_nin, 256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.projection(x)

class PixPro(nn.Module):
    def __init__(
        self,
        backbone,
        momentum=0.99,
        ppm_layers=1,
        ppm_gamma=0.2,
        downsampling=32
    ):
        super(PixPro, self).__init__()

        #create the encoder and momentum encoder
        self.encoder = Encoder(backbone)
        self.mom_encoder = deepcopy(self.encoder)

        #hardcoded: encoder outputs 256
        self.ppm = PPM(256, gamma=ppm_gamma, nlayers=ppm_layers)

        #copy parameters from the encoder to momentum encoder
        #and turn off gradients
        for param in self.mom_encoder.parameters():
            param.requires_grad = False

        self.grid_downsample = nn.AvgPool2d(downsampling, stride=downsampling)

        self.momentum = momentum
        #momentum = 1 − (1 − momentum) * (cos(pi * k / K) + 1) / 2 #k current step

    @torch.no_grad()
    def _update_mom_encoder(self):
        for param, mom_param in zip(self.encoder.parameters(), self.mom_encoder.parameters()):
            mom_param.data = mom_param.data * self.momentum + param.data * (1. - self.momentum)


    def forward(self, view1, view2, view1_grid, view2_grid):
        #pass each view through each encoder
        y1 = self.ppm(self.encoder(view1))
        y2 = self.ppm(self.encoder(view2))

        with torch.no_grad():
            #update mom encoder before forward
            self._update_mom_encoder()

            z1 = self.mom_encoder(view1)
            z2 = self.mom_encoder(view2)

        view1_grid = self.grid_downsample(view1_grid)
        view2_grid = self.grid_downsample(view2_grid)
        
        return y1, y2, z1, z2, view1_grid, view2_grid

def grid_distances(grid1, grid2):
    #grid: (B, 2, H, W) --> (B, 2, H * W)
    h, w = grid1.size()[-2:]
    grid1 = grid1.flatten(2, -1)[..., :, None] #(B, 2, H * W, 1)
    grid2 = grid2.flatten(2, -1)[..., None, :] #(B, 2, 1, H * W)

    y_distances = grid1[:, 0] - grid2[:, 0]
    x_distances = grid1[:, 1] - grid2[:, 1]

    return torch.sqrt(y_distances ** 2 + x_distances ** 2)

class ConsistencyLoss(nn.Module):
    def __init__(self, distance_thr=0.7):
        super(ConsistencyLoss, self).__init__()
        self.distance_thr = distance_thr
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, y1, y2, z1, z2, view1_grid, view2_grid):
        #(B, C, H * W)
        y1 = y1.flatten(2, -1)
        y2 = y2.flatten(2, -1)
        z1 = z1.flatten(2, -1)
        z2 = z2.flatten(2, -1)

        #pairwise distances between grid coordinates
        #(B, C, H * W, H * W)
        distances = grid_distances(view1_grid, view2_grid)

        #determine normalization factors for view1 and view2
        #(i.e. distance between "feature map bins")
        view1_bin = torch.norm(view1_grid[..., 1, 1] - view1_grid[..., 0, 0], dim=-1)
        view2_bin = torch.norm(view2_grid[..., 1, 1] - view2_grid[..., 0, 0], dim=-1)

        view1_distances = distances / view1_bin[:, None, None]
        view2_distances = distances / view2_bin[:, None, None]

        #average only over matched "pixels"
        view1_similarity = self.cosine_sim(y1[..., :, None], z2[..., None, :])
        view1_mask = view1_distances <= self.distance_thr
        view1_loss = view1_similarity.masked_select(view1_mask).mean()

        view2_similarity = self.cosine_sim(y2[..., :, None], z1[..., None, :])
        view2_mask = view2_distances <= self.distance_thr
        view2_loss = view2_similarity.masked_select(view2_mask).mean()

        return -1 * (view1_loss + view2_loss)
