import timm
import torch
from torch import nn


class Model(nn.Module):
    # def __init__(self, model_name="resnet50", in_chans=13):
    def __init__(self, model_name="resnet50"):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            # model_name, pretrained=True, in_chans=in_chans, drop_path_rate=0.2
            model_name, pretrained=True, drop_path_rate=0.2
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.model_name = model_name
        if self.model_name.startswith('resnet50'):
            self.frames_linear = nn.Sequential(nn.Linear(self.backbone.num_features, self.backbone.num_features), nn.ReLU())
        else:
            if self.model_name.startswith('convnext'):
                self.frames_linear = nn.Sequential(nn.Linear(self.backbone.num_features, self.backbone.num_features), nn.ReLU())
            else:
                self.frames_linear = nn.Sequential(nn.Linear(self.backbone.num_features * 2, self.backbone.num_features), nn.ReLU())
            self.aggregater = nn.Sequential(
                                nn.Conv3d(self.backbone.num_features, self.backbone.num_features, kernel_size=(3,1,1), stride=(2, 1, 1)), 
                                nn.MaxPool3d(kernel_size=(2,1,1), stride=(2, 1, 1))
                            )


        self.backbone.reset_classifier(0, "")
        self.mlp = nn.Sequential(
            # nn.Linear(130, 64),
            nn.Linear(18, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(64, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        # self.fc = nn.Linear(64 + self.backbone.num_features * 2, 1)
        self.fc = nn.Linear(64 + self.backbone.num_features, 1)

    def forward(self, images, features, labels=None):
        b, c, h, w = images.shape
        
        # images = images.reshape(b * 2, c // 2, h, w)
        # images = self.backbone(images).reshape(b, -1)

        images = images.reshape(b, c // 3, 3, h, w).flatten(0, 1)
        images = self.backbone(images)
        _, _c, _h, _w = images.shape
        if self.model_name == 'resnet50':
            images = images.reshape(b, c // 3, _c, _h, _w)
            images = images.mean(dim=1)
        else:
            images = images.reshape(b, _c, c // 3, _h, _w)
            images = self.aggregater(images)
        images = self.frames_linear(self.flatten(self.global_pool(images)))

        features = self.mlp(features)
        y = self.fc(torch.cat([images, features], dim=1))
        # _b, _f = images.shape
        # images = images.reshape(_b, _f//4, 4).mean(dim=-1)
        # return y, torch.cat([images, features], dim=1)
        return y
