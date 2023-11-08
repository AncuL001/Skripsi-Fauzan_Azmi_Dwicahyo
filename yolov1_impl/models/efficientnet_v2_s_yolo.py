from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch import nn

# TODO fill time per epoch on 1660 SUPER
def efficientnet_v2_s_yolo(split_size, num_boxes, num_classes, dropout_percentage=0.0) -> nn.Module:
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    n_inputs = model.classifier[1].in_features
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

    model.classifier = classifier
    return model
