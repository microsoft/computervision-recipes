from fastai.vision import *
from utils_ic.datasets import imagenet_labels
from utils_ic.constants import IMAGENET_IM_SIZE


def model_to_learner(
    model: nn.Module, im_size: int = IMAGENET_IM_SIZE
) -> Learner:
    """Create Learner based on pyTorch ImageNet model.

    Args:
        model (nn.Module): Base ImageNet model. E.g. models.resnet18()
        im_size (int): Image size the model will expect to have.

    Returns:
         Learner: a model trainer for prediction
    """

    # Currently, fast.ai api requires to pass a DataBunch to create a model trainer (learner).
    # To use the learner for prediction tasks without retraining, we have to pass an empty DataBunch.
    # single_from_classes is deprecated, but this is the easiest go-around method.
    # Create ImageNet data spec as an empty DataBunch.
    empty_data = ImageDataBunch.single_from_classes(
        "", classes=imagenet_labels(), size=im_size
    ).normalize(imagenet_stats)

    return Learner(empty_data, model)
