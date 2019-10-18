# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Adapted from https://github.com/feiyunzhang/i3d-non-local-pytorch/blob/master/transforms.py

import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch


class GroupScale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        cropped_imgs = [self.worker(img) for img in img_group]
        return cropped_imgs


class GroupRandomHorizontalFlip(object):

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupNormalize(object):
    
    def __init__(self, modality, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        self.modality = modality
        self.means = means
        self.stds = stds
        self.tensor_worker = torchvision.transforms.ToTensor()
        self.norm_worker = torchvision.transforms.Normalize(mean=means, std=stds)

    def __call__(self, img_group):
        if self.modality == "RGB":
            # Convert images to tensors in range [0, 1]
            img_tensors = [self.tensor_worker(img) for img in img_group]
            # Normalize to imagenet means and stds
            img_tensors = [self.norm_worker(img) for img in img_tensors]
        else:
            # Convert images to numpy arrays
            img_arrays = [np.asarray(img).transpose([2, 0, 1]) for img in img_group]
            # Scale to [-1, 1] and convert to tensor
            img_tensors = [torch.from_numpy((img / 255.) * 2 - 1) for img in img_arrays]
        return img_tensors


class Stack(object):

    def __call__(self, img_tensors):
        # Stack tensors and permute from D x C x H x W to C x D x H x W
        return torch.stack(img_tensors, dim=0).permute(1, 0, 2, 3).float()