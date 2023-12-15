from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torch import nn

# 5:04 minutes per epoch on 1660 SUPER
def vgg11_yolo(split_size, num_boxes, num_classes, dropout_percentage=0.0) -> nn.Module:
    model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
    n_inputs = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(n_inputs, 4096),
        nn.Dropout(dropout_percentage),
        nn.LeakyReLU(0.1),
        nn.Linear(
            4096,
            split_size * split_size * (num_classes + num_boxes * 5),
        ),
    )
    model.classifier = classifier
    return model
