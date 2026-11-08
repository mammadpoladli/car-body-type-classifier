import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


def get_model(num_classes=8, dropout_rate=0.4):
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    return model
