# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import copy
import math
from pathlib import Path
import warnings
from typing import Callable, Tuple, Union, List

import decord
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import Compose

from .references import transforms_video as transforms
from .references.functional_video import denormalize

from ..common.misc import Config
from ..common.gpu import num_devices, db_num_workers

Trans = Callable[[object, dict], Tuple[object, dict]]

DEFAULT_MEAN = (0.43216, 0.394666, 0.37645)
DEFAULT_STD = (0.22803, 0.22145, 0.216989)


class VideoRecord(object):
    """
    This class is used for parsing split-files where each row contains a path
    and a label:

    Ex:
    ```
    path/to/my/clip_1 3
    path/to/another/clip_2 32
    ```
    """

    def __init__(self, data: List[str]):
        """ Initialized a VideoRecord

        Ex.
        data = ["path/to/video.mp4", 2, "cooking"]

        Args:
            row: a list where first element is the path and second element is
            the label, and the third element (optional) is the label name
        """
        assert len(data) >= 2 and len(data) <= 3
        assert isinstance(data[0], str)
        assert isinstance(int(data[1]), int)
        if len(data) == 3:
            assert isinstance(data[2], str)

        self._data = data
        self._num_frames = None

    @property
    def path(self) -> str:
        return self._data[0]

    @property
    def num_frames(self) -> int:
        if self._num_frames is None:
            self._num_frames = int(
                len([x for x in Path(self._data[0]).glob("img_*")]) - 1
            )
        return self._num_frames

    @property
    def label(self) -> int:
        return int(self._data[1])

    @property
    def label_name(self) -> str:
        return None if len(self._data) <= 2 else self._data[2]


def get_transforms(train: bool = True, tfms_config: Config = None) -> Trans:
    """ Get default transformations to apply depending on whether we're applying it to the training or the validation set. If no tfms configurations are passed in, use the defaults.

    Args:
        train: whether or not this is for training
        tfms_config: Config object with tranforms-related configs

    Returns:
        A list of transforms to apply
    """
    if tfms_config is None:
        tfms_config = get_default_tfms_config(train=train)

    # 1. resize
    tfms = [
        transforms.ToTensorVideo(),
        transforms.ResizeVideo(
            tfms_config.im_scale, tfms_config.resize_keep_ratio
        ),
    ]

    # 2. crop
    if tfms_config.random_crop:
        if tfms_config.random_crop_scales:
            crop = transforms.RandomResizedCropVideo(
                tfms_config.input_size, tfms_config.random_crop_scales
            )
        else:
            crop = transforms.RandomCropVideo(tfms_config.input_size)
    else:
        crop = transforms.CenterCropVideo(tfms_config.input_size)
    tfms.append(crop)

    # 3. flip
    tfms.append(transforms.RandomHorizontalFlipVideo(tfms_config.flip_ratio))

    # 4. normalize
    tfms.append(transforms.NormalizeVideo(tfms_config.mean, tfms_config.std))

    return Compose(tfms)


def get_default_tfms_config(train: bool) -> Config:
    """
    Args:
        train: whether or not this is for training

    Settings:
        input_size (int or tuple): Model input image size.
        im_scale (int or tuple): Resize target size.
        resize_keep_ratio (bool): If True, keep the original ratio when resizing.
        mean (tuple): Normalization mean.
        if train:
        std (tuple): Normalization std.
        flip_ratio (float): Horizontal flip ratio.
        random_crop (bool): If False, do center-crop.
        random_crop_scales (tuple): Range of size of the origin size random cropped.
    """
    flip_ratio = 0.5 if train else 0.0
    random_crop = True if train else False
    random_crop_scales = (0.6, 1.0) if train else None

    return Config(
        dict(
            input_size=112,
            im_scale=128,
            resize_keep_ratio=True,
            mean=DEFAULT_MEAN,
            std=DEFAULT_STD,
            flip_ratio=flip_ratio,
            random_crop=random_crop,
            random_crop_scales=random_crop_scales,
        )
    )


class VideoDataset:
    """ A video recognition dataset. """

    def __init__(
        self,
        root: str,
        seed: int = None,
        train_pct: float = 0.75,
        num_samples: int = 1,
        sample_length: int = 8,
        sample_step: int = 1,
        temporal_jitter: bool = True,
        temporal_jitter_step: int = 2,
        random_shift: bool = True,
        batch_size: int = 8,
        video_ext: str = "mp4",
        warning: bool = False,
        train_split_file: str = None,
        test_split_file: str = None,
        train_transforms: Trans = get_transforms(train=True),
        test_transforms: Trans = get_transforms(train=False),
    ) -> None:
        """ initialize dataset

        Arg:
            root: Videos directory.
            seed: random seed
            train_pct: percentage of dataset to use for training
            num_samples: Number of clips to sample from each video.
            sample_length: Number of consecutive frames to sample from a video (i.e. clip length).
            sample_step: Sampling step.
            temporal_jitter: Randomly skip frames when sampling each frames.
            temporal_jitter_step: temporal jitter in frames
            random_shift: Random temporal shift when sample a clip.
            video_ext: Video file extension.
            warning: On or off warning.
            train_split_file: Annotation file containing video filenames and labels.
            test_split_file: Annotation file containing video filenames and labels.
            train_transforms: transforms for training
            test_transforms: transforms for testing
        """

        assert sample_step > 0
        assert num_samples > 0

        if temporal_jitter:
            assert temporal_jitter_step > 0

        if train_split_file:
            assert Path(train_split_file).exists()
            assert (
                test_split_file is not None and Path(test_split_file).exists()
            )

        if test_split_file:
            assert Path(test_split_file).exists()
            assert (
                train_split_file is not None
                and Path(train_split_file).exists()
            )

        self.root = root
        self.seed = seed
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.sample_step = sample_step
        self.presample_length = sample_length * sample_step
        self.temporal_jitter_step = temporal_jitter_step
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.random_shift = random_shift
        self.temporal_jitter = temporal_jitter
        self.batch_size = batch_size
        self.video_ext = video_ext
        self.warning = warning

        # create training and validation datasets
        self.train_ds, self.test_ds = (
            self.split_with_file(
                train_split_file=train_split_file,
                test_split_file=test_split_file,
            )
            if train_split_file
            else self.split_by_folder(train_pct=train_pct)
        )

        # initialize dataloaders
        self.init_data_loaders()

    def split_by_folder(
        self, train_pct: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        """ Split this dataset into a training and testing set based on the
        folders that the videos are in.

        ```
        /data
        +-- action_class_1
        |   +-- video_01.mp4
        |   +-- video_02.mp4
        |   +-- ...
        +-- action_class_2
        |   +-- video_11.mp4
        |   +-- video_12.mp4
        |   +-- ...
        +-- ...
        ```

        Args:
            train_pct: the ratio of images to use for training vs
            testing

        Return
            A training and testing dataset in that order
        """
        self.video_records = []

        # get all dirs in root (and make sure they are dirs)
        dirs = []
        for entry in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, entry)):
                dirs.append(os.path.join(self.root, entry))

        # add each video in each dir as a video record
        label = 0
        self.classes = []
        for action in dirs:
            action = os.path.basename(os.path.normpath(action))
            self.video_records.extend(
                [
                    VideoRecord(
                        [
                            os.path.join(self.root, action, vid.split(".")[0]),
                            label,
                            action,
                        ]
                    )
                    for vid in os.listdir(os.path.join(self.root, action))
                ]
            )
            label += 1
            self.classes.append(action)

        # random split
        test_num = math.floor(len(self) * (1 - train_pct))
        if self.seed:
            torch.manual_seed(self.seed)

        # set indices
        indices = torch.randperm(len(self)).tolist()
        train_range = indices[test_num:]
        test_range = indices[:test_num]

        return self.split_train_test(train_range, test_range)

    def split_with_file(
        self,
        train_split_file: Union[Path, str],
        test_split_file: Union[Path, str],
    ) -> Tuple[Dataset, Dataset]:
        """ Split this dataset into a training and testing set using a split file.

        Each line in the split file must use the form:
        ```
        path/to/jumping/video_name_1 3
        path/to/swimming/video_name_2 5
        path/to/another/jumping/video_name_3 3
        ```

        Args:
            split_files: a tuple of 2 files

        Return:
            A training and testing dataset in that order
        """
        self.video_records = []

        # add train records
        self.video_records.extend(
            [
                VideoRecord(row.strip().split(" "))
                for row in open(train_split_file)
            ]
        )
        train_len = len(self.video_records)

        # add validation records
        self.video_records.extend(
            [
                VideoRecord(row.strip().split(" "))
                for row in open(test_split_file)
            ]
        )

        # create indices
        indices = torch.arange(0, len(self.video_records))
        train_range = indices[:train_len]
        test_range = indices[train_len:]

        return self.split_train_test(train_range, test_range)

    def split_train_test(
        self, train_range: torch.Tensor, test_range: torch.Tensor,
    ) -> Tuple[Dataset, Dataset]:
        """ Split this dataset into a training and testing set

        Args:
            train_range: range of indices for training set
            test_range: range of indices for testing set

        Return
            A training and testing dataset in that order
        """
        # create train subset
        train = copy.deepcopy(Subset(self, train_range))
        train.dataset.transforms = self.train_transforms
        train.dataset.sample_step = (
            self.temporal_jitter_step
            if self.temporal_jitter
            else self.sample_step
        )
        train.dataset.presample_length = self.sample_length * self.sample_step

        # create test subset
        test = copy.deepcopy(Subset(self, test_range))
        test.dataset.transforms = self.test_transforms
        test.dataset.random_shift = False
        test.dataset.temporal_jitter = False

        return train, test

    def init_data_loaders(self) -> None:
        """ Create training and validation data loaders. """
        devices = num_devices()

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size * devices,
            shuffle=True,
            num_workers=db_num_workers(),
            pin_memory=True,
        )

        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=self.batch_size * devices,
            shuffle=False,
            num_workers=db_num_workers(),
            pin_memory=True,
        )

    def __len__(self) -> int:
        return len(self.video_records)

    def _sample_indices(self, record: VideoRecord) -> List[int]:
        """
        Create a list of frame-wise offsets into a video record. Depending on
        whether or not 'random shift' is used, perform a uniform sample or a
        random sample.

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
                        size=self.num_samples,
                    )
                )
            else:
                # Uniform sample
                distance = (
                    record.num_frames - self.presample_length + 1
                ) / self.num_samples
                offsets = np.array(
                    [
                        int(distance / 2.0 + distance * x)
                        for x in range(self.num_samples)
                    ]
                )
        else:
            if self.warning:
                warnings.warn(
                    f"num_samples and/or sample_length > num_frames in {record.path}"
                )
            offsets = np.zeros((self.num_samples,), dtype=int)

        return offsets

    def _get_frames(
        self, video_reader: decord.VideoReader, offset: int,
    ) -> List[np.ndarray]:
        """ Get frames at sample length.

        Args:
            video_reader: the decord tool for parsing videos
            offset: where to start the reader from

        Returns
            Frames at sample length in a List
        """
        clip = list()

        # decord.seek() seems to have a bug. use seek_accurate().
        video_reader.seek_accurate(offset)

        # first frame
        clip.append(video_reader.next().asnumpy())

        # remaining frames
        try:
            for i in range(self.sample_length - 1):
                step = (
                    randint(self.sample_step + 1)
                    if self.temporal_jitter
                    else self.sample_step
                )

                if step == 0 and self.temporal_jitter:
                    clip.append(clip[-1].copy())
                else:
                    if step > 1:
                        video_reader.skip_frames(step - 1)
                    cur_frame = video_reader.next().asnumpy()
                    clip.append(cur_frame)

        except StopIteration:
            # pass when video has ended
            pass

        # if clip needs more frames, simply duplicate the last frame in the clip.
        while len(clip) < self.sample_length:
            clip.append(clip[-1].copy())

        return clip

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, int]:
        """
        Return:
            (clips (torch.tensor), label (int))
        """
        record = self.video_records[idx]
        video_reader = decord.VideoReader(
            "{}.{}".format(
                os.path.join(self.root, record.path), self.video_ext
            ),
            # TODO try to add `ctx=decord.ndarray.gpu(0) or .cuda(0)`
        )
        record._num_frames = len(video_reader)

        offsets = self._sample_indices(record)
        clips = np.array([self._get_frames(video_reader, o) for o in offsets])

        if self.num_samples == 1:
            return (
                # [T, H, W, C] -> [C, T, H, W]
                self.transforms(torch.from_numpy(clips[0])),
                record.label,
            )

        else:
            return (
                # [S, T, H, W, C] -> [S, C, T, H, W]
                torch.stack(
                    [self.transforms(torch.from_numpy(c)) for c in clips]
                ),
                record.label,
            )

    def _show_batch(
        self,
        images: List[torch.tensor],
        labels: List[int],
        sample_length: int,
        mean: Tuple[int, int, int] = DEFAULT_MEAN,
        std: Tuple[int, int, int] = DEFAULT_STD,
    ) -> None:
        """
        Display a batch of images.

        Args:
            images: List of sample (clip) tensors
            labels: List of labels
            sample_length: Number of frames to show for each sample
            mean: Normalization mean
            std: Normalization std-dev
        """
        batch_size = len(images)
        plt.tight_layout()
        fig, axs = plt.subplots(
            batch_size,
            sample_length,
            figsize=(4 * sample_length, 3 * batch_size),
        )

        for i, ax in enumerate(axs):
            if batch_size == 1:
                clip = images[0]
            else:
                clip = images[i]
            clip = Rearrange("c t h w -> t c h w")(clip)
            if not isinstance(ax, np.ndarray):
                ax = [ax]
            for j, a in enumerate(ax):
                a.axis("off")
                a.imshow(
                    np.moveaxis(denormalize(clip[j], mean, std).numpy(), 0, -1)
                )

                # display label/label_name on the first image
                if j == 0:
                    a.text(
                        x=3,
                        y=15,
                        s=f"{labels[i]}",
                        fontsize=20,
                        bbox=dict(facecolor="white", alpha=0.80),
                    )

    def show_batch(self, train_or_test: str = "train", rows: int = 2) -> None:
        """Plot first few samples in the datasets"""
        if train_or_test == "train":
            batch = [self.train_ds[i] for i in range(rows)]
        elif train_or_test == "test":
            batch = [self.test_ds[i] for i in range(rows)]
        else:
            raise ValueError("Unknown data type {}".format(which_data))

        images = [im[0] for im in batch]
        labels = [im[1] for im in batch]

        self._show_batch(images, labels, self.sample_length)
