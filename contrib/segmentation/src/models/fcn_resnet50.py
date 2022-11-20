import torchvision
from torchvision.models.segmentation.fcn import FCNHead


def get_fcn_resnet50(n_classes, pretrained=True, is_feature_extracting: bool = False):
    """Load Fully Convolutional Network with ResNet-50 backbone

    Parameters
    ----------
    n_classes : int
        Number of classes
    pretrained : bool
        True if model should use pre-trained weights from COCO
    is_feature_extracting : bool
        True if the convolutional layers should be set to non-trainable retaining their original
        parameters
    """
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained)

    if is_feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = FCNHead(2048, n_classes)
    model.aux_classifier = FCNHead(1024, n_classes)

    return model
