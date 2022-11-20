from typing import Tuple
import albumentations as A


def _preprocessing(patch_dim: Tuple[int, int] = (512, 512)):
    transform = A.Compose(
        [
            # This allows meaningful yet stochastic cropped views
            A.CropNonEmptyMaskIfExists(patch_dim[0], patch_dim[1], p=1),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.25),
            A.ColorJitter(p=0.25),
            A.GaussNoise(p=0.25),
            A.CoarseDropout(p=0.5, max_holes=64, max_height=8, max_width=8),
            A.RandomBrightnessContrast(p=0.25),
        ],
    )
    return transform


def _augmentation(patch_dim: Tuple[int, int] = (512, 512)):
    transform = A.Compose(
        [
            # This allows meaningful yet stochastic cropped views
            A.CropNonEmptyMaskIfExists(patch_dim[0], patch_dim[1], p=1),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.25),
            A.ColorJitter(p=0.25),
            A.GaussNoise(p=0.25),
            A.CoarseDropout(p=0.5, max_holes=64, max_height=8, max_width=8),
            A.RandomBrightnessContrast(p=0.25),
        ],
    )
    return transform


preprocessing = _preprocessing()
augmentation = _augmentation()
