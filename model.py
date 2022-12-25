import torch
from torch import nn
import timm


class Model(nn.Module):
    def __init__(self, model_name="resnet50", in_chans=13):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=500, in_chans=in_chans
        )
        self.mlp = nn.Sequential(
            nn.Linear(18, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(64, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc = nn.Linear(64 + 500 * 2, 1)

    def forward(self, images, features, labels=None):
        b, c, h, w = images.shape
        images = images.reshape(b * 2, c // 2, h, w)
        images = self.backbone(images).reshape(b, -1)
        features = self.mlp(features)
        y = self.fc(torch.cat([images, features], dim=1))
        return y
