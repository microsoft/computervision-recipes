# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Adapted from https://github.com/feiyunzhang/i3d-non-local-pytorch/blob/master/dataset.py

import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from pathlib import Path

import torchvision
from torchvision import datasets, transforms
from videotransforms import (
    GroupRandomCrop, GroupRandomHorizontalFlip,
    GroupScale, GroupCenterCrop, GroupNormalize, Stack
)

from itertools import cycle


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(
            len([x for x in Path(
                self._data[0]).glob('img_*')])-1)

    @property
    def label(self):
        return int(self._data[1])


class I3DDataSet(data.Dataset):
    def __init__(self, data_root, split=1, sample_frames=64, 
            modality='RGB', transform=lambda x:x,
            train_mode=True, sample_frames_at_test=False):

        self.data_root = data_root
        self.split = split
        self.sample_frames = sample_frames
        self.modality = modality
        self.transform = transform
        self.train_mode = train_mode
        self.sample_frames_at_test = sample_frames_at_test

        self._parse_split_files()


    def _parse_split_files(self):
            # class labels assigned by sorting the file names in /data/hmdb51_splits directory
            file_list = sorted(Path('./data/hmdb51_splits').glob('*'+str(self.split)+'.txt'))
            video_list = []
            for class_idx, f in enumerate(file_list):
                class_name = str(f).strip().split('/')[2][:-16]
                for line in open(f):
                    tokens = line.strip().split(' ')
                    video_path = self.data_root+class_name+'/'+tokens[0][:-4]
                    record = (video_path, class_idx)
                    # 1 indicates video should be in training set
                    if self.train_mode & (tokens[-1] == '1'):
                        video_list.append(VideoRecord(record))
                    # 2 indicates video should be in test set
                    elif (self.train_mode == False) & (tokens[-1] == '2'):
                        video_list.append(VideoRecord(record))
                
            self.video_list = video_list


    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            img_path = os.path.join(directory, 'img_{:05}.jpg'.format(idx))
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
            return img
        else:
            try:
                img_path = os.path.join(directory, 'flow_x_{:05}.jpg'.format(idx))
                x_img = Image.open(img_path).convert('L')
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
            try:
                img_path = os.path.join(directory, 'flow_y_{:05}.jpg'.format(idx))
                y_img = Image.open(img_path).convert('L')
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
            # Combine flow images into single PIL image
            x_img = np.array(x_img, dtype=np.float32)
            y_img = np.array(y_img, dtype=np.float32)
            img = np.asarray([x_img, y_img]).transpose([1, 2, 0])
            img = Image.fromarray(img.astype('uint8'))
            return img


    def _sample_indices(self, record):
        if record.num_frames > self.sample_frames:
            start_pos = randint(record.num_frames - self.sample_frames + 1)
            indices = range(start_pos, start_pos + self.sample_frames, 1)
        else:
            indices = [x for x in range(record.num_frames)]
        if len(indices) < self.sample_frames:
            self._loop_indices(indices)
        return indices


    def _loop_indices(self, indices):
        indices_cycle = cycle(indices)
        while len(indices) < self.sample_frames:
            indices.append(next(indices_cycle))


    def __getitem__(self, index):
        record = self.video_list[index]
        # Sample frames from the the video for training, or if sampling
        # turned on at test time
        if self.train_mode or self.sample_frames_at_test:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = [i for i in range(record.num_frames)]
        # Image files are 1-indexed
        segment_indices = [i+1 for i in segment_indices]
        # Get video frame images
        images = []
        for i in segment_indices:
            seg_img = self._load_image(record.path, i)
            if seg_img is None:
                raise ValueError("Couldn't load", record.path, i)
            images.append(seg_img)
        # Apply transformations
        transformed_images = self.transform(images)

        return transformed_images, record.label


    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':

    input_size = 224
    resize_small_edge = 256

    train_rgb = I3DDataSet(
        data_root='/datadir/rawframes/',
        split=1,
        sample_frames = 64,
        modality='RGB',
        train_mode=True,
        sample_frames_at_test=False,
        transform=torchvision.transforms.Compose([
            GroupScale(resize_small_edge),
            GroupRandomCrop(input_size),
            GroupRandomHorizontalFlip(),
            GroupNormalize(modality="RGB"),
            Stack(),
        ])
    )
    item = train_rgb.__getitem__(10)
    print("train_rgb:")
    print(item[0].size())
    print("max=", item[0].max())
    print("min=", item[0].min())
    print("label=",item[1])

    val_rgb = I3DDataSet(
        data_root='/datadir/rawframes/',
        split=1,
        sample_frames = 64,
        modality='RGB',
        train_mode=False,
        sample_frames_at_test=False,
        transform=torchvision.transforms.Compose([
            GroupScale(resize_small_edge),
            GroupCenterCrop(input_size),
            GroupNormalize(modality="RGB"),
            Stack(),
        ])
    )
    item = val_rgb.__getitem__(10)
    print("val_rgb:")
    print(item[0].size())
    print("max=", item[0].max())
    print("min=", item[0].min())
    print("label=",item[1])

    train_flow = I3DDataSet(
        data_root='/datadir/rawframes/',
        split=1,
        sample_frames = 64,
        modality='flow',
        train_mode=True,
        sample_frames_at_test=False,
        transform=torchvision.transforms.Compose([
            GroupScale(resize_small_edge),
            GroupRandomCrop(input_size),
            GroupRandomHorizontalFlip(),
            GroupNormalize(modality="flow"),
            Stack(),
        ])
    )
    item = train_flow.__getitem__(100)
    print("train_flow:")
    print(item[0].size())
    print("max=", item[0].max())
    print("min=", item[0].min())
    print("label=",item[1])

    val_flow = I3DDataSet(
        data_root='/datadir/rawframes/',
        split=1,
        sample_frames = 64,
        modality='flow',
        train_mode=False,
        sample_frames_at_test=False,
        transform=torchvision.transforms.Compose([
            GroupScale(resize_small_edge),
            GroupCenterCrop(input_size),
            GroupNormalize(modality="flow"),
            Stack(),
        ])
    )
    item = val_flow.__getitem__(100)
    print("val_flow:")
    print(item[0].size())
    print("max=", item[0].max())
    print("min=", item[0].min())
    print("label=",item[1])