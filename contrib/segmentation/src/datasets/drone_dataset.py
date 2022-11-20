from glob import glob
from os.path import join
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple
from src.datasets.coco import BoundingBox

from torch.utils.data import Dataset


class DroneDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, class_dict_path: str):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.images_index = [
            filename.split(".")[0] for filename in glob("*.jpg")
        ]

        class_dict = pd.read_csv(class_dict_path).to_dict("index")
        self.class_id_to_name = {
            class_id: rec["name"] for class_id, rec in class_dict.items()
        }
        self.rgb_to_class = {
            (rec["r"], rec["g"], rec["b"]): int(class_id)
            for class_id, rec in class_dict.items()
        }

    def _mask_rgb_to_class_label(self, rgb_mask: np.ndarray):
        """The Semantic Drone Dataset formats their masks as an RGB mask
        To prepare the mask for use with a PyTorch model, we must encode
        the mask as a 2D array of class labels

        Parameters
        ----------
        rgb_mask : np.ndarray
            Mask array with RGB values for each class

        Returns
        -------
        mask : np.ndarray
            Mask with shape `(height, width)` with class_id values where they occur
        """
        height, width, _ = rgb_mask.shape
        mask = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                mask[i][j] = self.rgb_to_class[tuple(rgb_mask[i][j])]
        return mask

    def __getitem__(
        self, image_id: int
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        filename = self.images_index[image_id]
        image_filepath = join(self.images_dir, f"{filename}.jpg")
        image = Image.open(image_filepath).convert("RGB")
        image = np.array(image).astype("float32")

        mask_filepath = join(self.images_dir, f"{filename}.png")
        mask = Image.open(mask_filepath).convert("RGB")
        mask = np.array(mask).astype("uint8")

        mask = self._mask_rgb_to_class_label(mask)

        return image, [], mask, []
