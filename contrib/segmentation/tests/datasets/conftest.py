import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def high_resolution_image() -> Image.Image:
    height = 3632
    width = 5456
    channels = 3
    image: np.ndarray = np.random.randint(
        low=0, high=256, size=height * width * channels, dtype=np.uint8
    )
    image = image.reshape((height, width, channels))
    return Image.fromarray(image)
