from fastai.vision import models
from utils_ic.imagenet_models import load_learner


def test_load_learner():
    # Test if the function loads an ImageNet model (ResNet) trainer
    learn = load_learner(models.resnet34(pretrained=True))
    assert len(learn.data.classes) == 1000  # Check Image net classes
    assert isinstance(learn.model, models.ResNet)

    # Test with SqueezeNet
    learn = load_learner(models.squeezenet1_0())
    assert len(learn.data.classes) == 1000
    assert isinstance(learn.model, models.SqueezeNet)
