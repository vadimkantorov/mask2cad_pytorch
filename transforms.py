import torch
import torch.nn as nn

import torchvision.transforms.functional
import torchvision.transforms.transforms

class Augmentations(nn.Module):
    def forward(self, images, targets, views):
        # images <- hflips, rescale?
        # renderings <- HSV-space jittering, random crop, random resize
        pass
