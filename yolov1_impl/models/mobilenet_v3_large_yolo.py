from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch import nn

# TODO fill time per epoch on 1660 SUPER
def mobilenet_v3_large_yolo(split_size, num_boxes, num_classes) -> nn.Module:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    n_inputs = model.classifier[0].in_features
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

    model.classifier = classifier
    return model
