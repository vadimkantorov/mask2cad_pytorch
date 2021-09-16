import random
import torch
import torchvision

import torch.nn as nn
import torchvision.transforms.functional as Fv
import torchvision.transforms.transforms as T

#scale_factor = min(self.target_image_size[0] / img.shape[-1], self.target_image_size[1] / img.shape[-2])
#img = F.interpolate(img.unsqueeze(0), self.target_image_size).squeeze(0) if img.numel() > 0 else torch.empty((0, self.target_image_size[1], self.target_image_size[0]), dtype = torch.uint8)
#mask = F.interpolate(mask.unsqueeze(0), self.target_image_size).squeeze(0) if mask.numel() > 0 else torch.empty((0, self.target_image_size[1], self.target_image_size[0]), dtype = torch.uint8)
##img = F.interpolate(img.unsqueeze(0), scale_factor = scale_factor).squeeze(0)
##mask = F.interpolate(img.unsqueeze(0), scale_factor = scale_factor).squeeze(0)
#bbox = [bbox[0] * scale_factor, bbox[1] * scale_factor, bbox[2] * scale_factor, bbox[3] * scale_factor]

class MaskRCNNAugmentations(nn.Module):
    # https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
    # _C.INPUT.MIN_SIZE_TRAIN = (800,)
    def __init__(self, p = 0.5, short_edge_length = (640, 672, 704, 736, 768, 800), max_size = 1333):
        super().__init__()
        # https://github.com/facebookresearch/detectron2/blob/main/configs/common/data/coco.py
		self.transforms = [ResizeShortestEdge(short_edge_length = short_edge_length, max_size = max_size), RandomHorizontalFlip(p = p)]
		# L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),

    def forward(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Mask2CADAugmentations(nn.Module):
    def __init__(self, shape_view_side_size = 128):
        super().__init__()
        self.shape_view_transforms = [RandomPhotometricDistort(), T.RandomCrop(shape_view_side_size)]

    def forward(self, image, target):
        for t in self.shape_view_transforms:
            target['shape_views'] = t(target['shape_views'])
        return image, target

class ResizeShortestEdge(nn.Module):
    #Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    #If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.

    def __init__(self, short_edge_length, max_size=sys.maxsize, interp='bilinear'):
        super().__init__()
		self.short_edge_length = short_edge_length
		self.max_size = max_size
		self.interp = interp

    def forward(self, image, target):
        h, w = image.shape[-2:]
		size = random.choice(self.short_edge_length)

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

        image = F.interpolate(img.unsqueeze(0), (new_h, new_w), mode = self.interp, align_corners = None if self.interp == "nearest" else False).squeeze(0)
		if 'boxes' in target:
			target['boxes'][..., 0::2] *= new_w / w
			target['boxes'][..., 1::2] *= new_h / h
		if 'masks' in target:        
			target['masks'] = F.interpolate(target['masks'], (new_h, new_w), mode='nearest', align_corners = None )

		return image, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image, target):
        if torch.rand(1) < self.p:
            image = Fv.hflip(image)
            if target is not None:
                width, _ = Fv.get_image_size(image)
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
            channels = Fv.get_image_num_channels(image)
            permutation = torch.randperm(channels)

            image = image[..., permutation, :, :]

        return image, target
