import timm
import torch
from torch import nn
from torchvision.ops import RoIAlign


class Model(nn.Module):
    def __init__(self, model_name="resnet50", in_chans=15):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, in_chans=in_chans, drop_path_rate=0.2
        )
        self.backbone.reset_classifier(0, "")
        self.box_roi_pool = RoIAlign(
            output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, aligned=True
        )
        self.box_head = nn.Sequential(
            nn.Linear(7 * 7 * self.backbone.num_features, 512), nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(130, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(64 + 512 * 2, 1)

    def forward(self, inputs):
        # Endzone
        feats0 = self.backbone(inputs["images0"])
        feats0 = self.box_roi_pool(feats0, inputs["boxes0"])
        feats0 = self.box_head(feats0.flatten(1, -1))
        # Sideline
        feats1 = self.backbone(inputs["images1"])
        feats1 = self.box_roi_pool(feats1, inputs["boxes1"])
        feats1 = self.box_head(feats1.flatten(1, -1))

        visual_feats = torch.cat([feats0, feats1], 1)
        track_feats = self.mlp(inputs["features"])
        logits = self.fc(torch.cat([visual_feats, track_feats], 1))
        return logits
