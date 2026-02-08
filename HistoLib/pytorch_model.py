import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def get_model(num_classes, freeze_base=False):
    """
    Creates a ResNet50 model with a custom classification head.
    Matches the architecture of the Keras model: RecNet50 base -> Dense(256) -> Dropout(0.5) -> Dense(num_classes).
    """
    # Load pre-trained ResNet50
    # Using default weights (IMAGENET1K_V2 is the most up-to-date for ResNet50 in torchvision)
    weights = ResNet50_Weights.DEFAULT 
    model = models.resnet50(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # The inputs to the fc layer in ResNet50 are 2048
    in_features = model.fc.in_features

    # Replace the FC layer with our custom head
    # Keras model: Dense(256, relu) -> Dropout(0.5) -> Dense(num_classes, softmax)
    # PyTorch CrossEntropyLoss expects logits, so no Softmax here.
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    return model
