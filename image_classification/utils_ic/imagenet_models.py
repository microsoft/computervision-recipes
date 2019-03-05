from fastai.vision import *
from utils_ic.datasets import imagenet_labels


# desired input image size for the ImageNet models
IM_SIZE = 224


def load_learner(model: nn.Module) -> Learner:
    """Load an ImageNet model trainer for prediction
    Args:
        model (nn.Module): Base model. E.g. models.resnet18()

    Returns:
         Learner: a model trainer
    """
    labels = imagenet_labels()

    # Currently, fast.ai api requires to pass a DataBunch to create a model trainer (learner).
    # To use the learner for prediction tasks without retraining, we have to pass an empty DataBunch.
    # single_from_classes is deprecated, but this is the easiest go-around method.
    empty_data = ImageDataBunch.single_from_classes(
        "", classes=labels, size=IM_SIZE
    ).normalize(imagenet_stats)

    return Learner(empty_data, model)
