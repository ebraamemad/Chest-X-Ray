import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

