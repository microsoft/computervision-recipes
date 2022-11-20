import numpy as np
import pytest
from torch.utils.data.dataset import Dataset


class SemanticSegmentationTestDataset(Dataset):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __getitem__(self, idx):
        image: np.ndarray = np.zeros((self.height, self.width, 3))
        mask: np.ndarray = np.zeros((self.height, self.width))

        mask[: self.height // 2, : self.width // 2] = 4

        return image, mask

    def __len__(self):
        return 4


@pytest.fixture
def semantic_segmentation_dataset():
    return SemanticSegmentationTestDataset(256, 256)
