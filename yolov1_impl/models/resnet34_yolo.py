from torchvision.models import resnet34, ResNet34_Weights
from torch import nn

# 1:04 minutes per epoch on 1660 SUPER
def resnet34_yolo(split_size, num_boxes, num_classes) -> nn.Module:
    model = resnet34(weights=ResNet34_Weights.DEFAULT)

    n_inputs = model.fc.in_features
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_inputs, 4096),
        nn.Dropout(0.0),
        nn.LeakyReLU(0.1),
        nn.Linear(
            4096,
            split_size * split_size * (num_classes + num_boxes * 5),
        ),
    )

    model.fc = classifier
    return model
