import math
import random
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.transforms as T

class MaskRCNNAugmentations(nn.Sequential):
    # https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
    # _C.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    # short_edge_length = (800,), max_size = 1333
    def __init__(self, p = 0.5, short_edge_length = 480, max_size = 640, noise_scale = 0.025):
        # https://github.com/facebookresearch/detectron2/blob/main/configs/common/data/coco.py
        super().__init__(ResizeShortestEdge(short_edge_length = short_edge_length, max_size = max_size), JitterBoxes(noise_scale = noise_scale), RandomHorizontalFlip(p = p))

    def forward(self, image, target):
        for t in self:
            image, target = t(image, target)
        return image, target

class Mask2CADAugmentations(nn.Sequential):
    def __init__(self, shape_view_side_size = 128):
        super().__init__(RandomPhotometricDistort(), T.RandomCrop(shape_view_side_size))

    def forward(self, image, target):
        for t in self:
            target['shape_views'] = t(target['shape_views'])
        return image, target

class JitterBoxes(nn.Module):
    # https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/box_utils.py
    def __init__(self, noise_scale = 0.025):
        super().__init__()
        self.noise_scale = noise_scale

    def forward(self, image, target):
        xmin, ymin, xmax, ymax = target['boxes'].unbind(dim = -1)
        xm, ym, xp, yp = (xmax - xmin), (ymax - ymin), (xmin + xmax), (ymin + ymax)
        j = torch.randn_like(boxes) * noise_scale
        new_cx, new_cy, new_w_half, new_h_half = (xp / 2.0 + j[..., 0] * w), (yp / 2.0 + j[..., 1] * h), 0.5 * xm * j[..., 2].exp(), 0.5 * ym * j[..., 3].exp()
        new_boxes = torch.stack([new_cx - new_w_half, new_cy - new_h_half, new_cx + new_w_half, new_cy + new_h_half], dim = -1)
        target['boxes'] = new_boxes
        return image, target

class ResizeShortestEdge(nn.Module):
    #Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    #If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.

    def __init__(self, short_edge_length, max_size=int(1e9), interp='bilinear'):
        super().__init__()
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.interp = interp

    def forward(self, image, target):
        h, w = image.shape[-2:]
        size = self.short_edge_length

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        image = F.interpolate(image.unsqueeze(0), (newh, neww), mode = self.interp, align_corners = None if self.interp == "nearest" else False).squeeze(0)
        target['image_height_width_resized'] = (newh, neww)
        if 'boxes' in target:
            target['boxes'][..., 0::2] *= neww / w
            target['boxes'][..., 1::2] *= newh / h
        if 'masks' in target:        
            target['masks'] = F.interpolate(target['masks'].to(torch.uint8), (newh, neww), mode='nearest', align_corners = None ).to(torch.bool)

        return image, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image, target):
        if torch.rand(1) < self.p:
            image = image.flip(-1)
            if target is not None:
                width = image.shape[-1]
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
        return image, target

class RandomPhotometricDistort(nn.Module):
    def __init__(self, contrast = (0.5, 1.5), saturation = (0.5, 1.5),
                 hue = (-0.05, 0.05), brightness = (0.875, 1.125), p = 0.5):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(self, image, target = None):
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels = image.shape[-3]
            permutation = torch.randperm(channels)

            image = image[..., permutation, :, :]

        return image, target
