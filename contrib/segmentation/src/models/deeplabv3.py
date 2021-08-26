import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def get_deeplabv3(n_classes: int, pretrained: bool = False, is_feature_extracting: bool = False):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=pretrained, aux_loss=True
    )

    if is_feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = DeepLabHead(2048, n_classes)
    model.aux_classifier = DeepLabHead(1024, n_classes)

    return model