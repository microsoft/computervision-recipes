from torch.utils.data.dataset import Dataset

from src.losses.loss import semantic_segmentation_class_balancer


def test_semantic_segmentation_class_balancer(semantic_segmentation_dataset: Dataset):
    weights = semantic_segmentation_class_balancer(semantic_segmentation_dataset)

    assert len(weights) == 5
    assert weights[0] == 0.25
    assert weights[4] == 0.75
