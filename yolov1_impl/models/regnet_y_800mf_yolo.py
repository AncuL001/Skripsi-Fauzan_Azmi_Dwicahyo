from torchvision.models import regnet_y_800mf, RegNet_Y_800MF_Weights
from torch import nn

# TODO 1:07 per epoch on 1660 SUPER
def regnet_y_800mf_yolo(split_size, num_boxes, num_classes, dropout_percentage=0.0) -> nn.Module:
    model = regnet_y_800mf(weights=RegNet_Y_800MF_Weights.DEFAULT)

    n_inputs = model.fc.in_features
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_inputs, 4096),
        nn.Dropout(dropout_percentage),
        nn.LeakyReLU(0.1),
        nn.Linear(
            4096,
            split_size * split_size * (num_classes + num_boxes * 5),
        ),
    )

    model.fc = classifier
    return model
