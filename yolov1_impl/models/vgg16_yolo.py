from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torch import nn


def vgg16_yolo(split_size, num_boxes, num_classes) -> nn.Module:
    model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    n_inputs = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(n_inputs, 4096),
        nn.Dropout(0.0),
        nn.LeakyReLU(0.1),
        nn.Linear(
            4096,
            split_size * split_size * (num_classes + num_boxes * 5),
        ),
    )
    model.classifier = classifier
    return model