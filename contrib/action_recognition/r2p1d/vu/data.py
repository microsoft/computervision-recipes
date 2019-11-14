# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from urllib.request import urlretrieve
import warnings

import decord
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .utils import transforms_video as transforms
from .utils.functional_video import denormalize


DEFAULT_MEAN = (0.43216, 0.394666, 0.37645)
DEFAULT_STD = (0.22803, 0.22145, 0.216989)


class _DatasetSpec:
    def __init__(self, label_url, root, num_classes):
        self.label_url = label_url
        self.root = root
        self.num_classes = num_classes
        self._class_names = None

    @property
    def class_names(self):
        if self._class_names is None:
            label_filepath = os.path.join(self.root, "label_map.txt")
            if not os.path.isfile(label_filepath):
                os.makedirs(self.root, exist_ok=True)
                urlretrieve(self.label_url, label_filepath)
            with open(label_filepath) as f:
                self._class_names = [l.strip() for l in f]
            assert len(self._class_names) == self.num_classes

        return self._class_names


KINETICS = _DatasetSpec(
    "https://github.com/microsoft/ComputerVision/files/3746975/kinetics400_lable_map.txt",
    os.path.join("data", "kinetics400"),
    400
)

HMDB51 = _DatasetSpec(
    "https://github.com/microsoft/ComputerVision/files/3746963/hmdb51_label_map.txt",
    os.path.join("data", "hmdb51"),
    51
)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row
        self._num_frames = -1

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        if self._num_frames == -1:
            self._num_frames = int(len([x for x in Path(self._data[0]).glob('img_*')]) - 1)
        return self._num_frames

    @property
    def label(self):
        return int(self._data[1])


class VideoDataset(Dataset):
    """
    Args:
        split_file (str): Annotation file containing video filenames and labels.
        video_dir (str): Videos directory.
        num_segments (int): Number of clips to sample from each video.
        sample_length (int): Number of consecutive frames to sample from a video (i.e. clip length).
        sample_step (int): Sampling step.
        input_size (int or tuple): Model input image size.
        im_scale (int or tuple): Resize target size.
        resize_keep_ratio (bool): If True, keep the original ratio when resizing.
        mean (tuple): Normalization mean.
        std (tuple): Normalization std.
        random_shift (bool): Random temporal shift when sample a clip.
        temporal_jitter (bool): Randomly skip frames when sampling each frames.
        flip_ratio (float): Horizontal flip ratio.
        random_crop (bool): If False, do center-crop.
        random_crop_scales (tuple): Range of size of the origin size random cropped.
        video_ext (str): Video file extension.
        warning (bool): On or off warning.
    """
    def __init__(
        self,
        split_file,
        video_dir,
        num_segments=1,
        sample_length=8,
        sample_step=1,
        input_size=112,
        im_scale=128,
        resize_keep_ratio=True,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        random_shift=False,
        temporal_jitter=False,
        flip_ratio=0.5,
        random_crop=False,
        random_crop_scales=(0.6, 1.0),
        video_ext="mp4",
        warning=False,
    ):
        # TODO maybe check wrong arguments to early failure
        assert sample_step > 0
        assert num_segments > 0

        self.video_dir = video_dir
        self.video_records = [
            VideoRecord(x.strip().split(" ")) for x in open(split_file)
        ]

        self.num_segments = num_segments
        self.sample_length = sample_length
        self.sample_step = sample_step
        self.presample_length = sample_length * sample_step

        # Temporal noise
        self.random_shift = random_shift
        self.temporal_jitter = temporal_jitter

        # Video transforms
        # 1. resize
        trfms = [
            transforms.ToTensorVideo(),
            transforms.ResizeVideo(im_scale, resize_keep_ratio),
        ]
        # 2. crop
        if random_crop:
            if random_crop_scales is not None:
                crop = transforms.RandomResizedCropVideo(input_size, random_crop_scales)
            else:
                crop = transforms.RandomCropVideo(input_size)
        else:
            crop = transforms.CenterCropVideo(input_size)
        trfms.append(crop)
        # 3. flip
        trfms.append(transforms.RandomHorizontalFlipVideo(flip_ratio))
        # 4. normalize
        trfms.append(transforms.NormalizeVideo(mean, std))
        self.transforms = Compose(trfms)
        self.video_ext = video_ext
        self.warning = warning

    def __len__(self):
        return len(self.video_records)

    def _sample_indices(self, record):
        """
        Args:
            record (VideoRecord): A video record.
        Return:
            list: Segment offsets (start indices)
        """
        if record.num_frames > self.presample_length:
            if self.random_shift:
                # Random sample
                offsets = np.sort(
                    randint(
                        record.num_frames - self.presample_length + 1,
                        size=self.num_segments,
                    )
                )
            else:
                # Uniform sample
                distance = (record.num_frames - self.presample_length + 1) / self.num_segments
                offsets = np.array(
                    [int(distance / 2.0 + distance * x) for x in range(self.num_segments)]
                )
        else:
            if self.warning:
                warnings.warn(
                    "num_segments and/or sample_length > num_frames in {}".format(
                        record.path
                    )
                )
            offsets = np.zeros((self.num_segments,), dtype=int)

        return offsets

    def _get_frames(self, video_reader, offset):
        clip = list()

        # decord.seek() seems to have a bug. use seek_accurate().
        video_reader.seek_accurate(offset)
        # first frame
        clip.append(video_reader.next().asnumpy())
        # remaining frames
        try:
            if self.temporal_jitter:
                for i in range(self.sample_length - 1):
                    step = randint(self.sample_step + 1)
                    if step == 0:
                        clip.append(clip[-1].copy())
                    else:
                        if step > 1:
                            video_reader.skip_frames(step - 1)
                        cur_frame = video_reader.next().asnumpy()
                        if len(cur_frame.shape) != 3:
                            # maybe end of the video
                            break
                        clip.append(cur_frame)
            else:
                for i in range(self.sample_length - 1):
                    if self.sample_step > 1:
                        video_reader.skip_frames(self.sample_step - 1)
                    cur_frame = video_reader.next().asnumpy()
                    if len(cur_frame.shape) != 3:
                        # maybe end of the video
                        break
                    clip.append(cur_frame)
        except StopIteration:
            pass

        # if clip needs more frames, simply duplicate the last frame in the clip.
        while len(clip) < self.sample_length:
            clip.append(clip[-1].copy())
                
        return clip

    def __getitem__(self, idx):
        """
        Return:
            clips (torch.tensor), label (int)
        """
        record = self.video_records[idx]
        video_reader = decord.VideoReader(
            "{}.{}".format(os.path.join(self.video_dir, record.path), self.video_ext),
            # TODO try to add `ctx=decord.ndarray.gpu(0) or .cuda(0)`
        )
        record._num_frames = len(video_reader)

        offsets = self._sample_indices(record)
        clips = np.array([self._get_frames(video_reader, o) for o in offsets])

        if self.num_segments == 1:
            # [T, H, W, C] -> [C, T, H, W]
            return self.transforms(torch.from_numpy(clips[0])), record.label
        else:
            # [S, T, H, W, C] -> [S, C, T, H, W]
            return (
                torch.stack([
                    self.transforms(torch.from_numpy(c)) for c in clips
                ]),
                record.label
            )


def show_batch(batch, sample_length, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """
    Args:
        batch (list[torch.tensor]): List of sample (clip) tensors
        sample_length (int): Number of frames to show for each sample
        mean (tuple): Normalization mean
        std (tuple): Normalization std-dev
    """
    batch_size = len(batch)
    plt.tight_layout()
    fig, axs = plt.subplots(
        batch_size,
        sample_length,
        figsize=(4 * sample_length, 3 * batch_size)
    )

    for i, ax in enumerate(axs):
        if batch_size == 1:
            clip = batch[0]
        else:
            clip = batch[i]
        clip = Rearrange("c t h w -> t c h w")(clip)
        if not isinstance(ax, np.ndarray):
            ax = [ax]
        for j, a in enumerate(ax):
            a.axis("off")
            a.imshow(
                np.moveaxis(
                    denormalize(
                        clip[j],
                        mean,
                        std,
                    ).numpy(),
                    0,
                    -1,
                )
            )