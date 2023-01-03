import timm
import torch
from torch import nn
from torchvision.ops import RoIAlign


class Model(nn.Module):
    def __init__(self, model_name="resnet50"):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            # in_chans=1,
            drop_path_rate=0.2,
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.frames_linear = nn.Sequential(nn.Linear(self.backbone.num_features, 512), nn.ReLU())

        self.backbone.reset_classifier(0, "")
        self.box_roi_pool = RoIAlign(
            output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, aligned=True
        )
        self.box_head = nn.Sequential(
            nn.Linear(7 * 7 * self.backbone.num_features, 512), nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(130, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            # nn.Dropout(0.2),
        )
        self.fc = nn.Linear(512 + 512 * 2 + 512 * 2, 1)

    def forward(self, inputs):
        B, T, H, W = inputs["images0"].shape
        images0 = inputs["images0"].reshape(B, T // 3, 3, H, W).flatten(0, 1)
        images1 = inputs["images1"].reshape(B, T // 3, 3, H, W).flatten(0, 1)

        # Endzone
        raw_feats0 = self.backbone(images0)
        _, c, h, w = raw_feats0.shape
        raw_feats0 = raw_feats0.reshape(B, T // 3, c, h, w)
        raw_feats0 = raw_feats0.mean(dim=1)
        feats0 = self.box_roi_pool(raw_feats0, inputs["boxes0"])
        feats0 = self.box_head(feats0.flatten(1, -1))
        # Sideline
        raw_feats1 = self.backbone(images1)
        raw_feats1 = raw_feats1.reshape(B, T // 3, c, h, w)
        raw_feats1 = raw_feats1.mean(dim=1)
        feats1 = self.box_roi_pool(raw_feats1, inputs["boxes1"])
        feats1 = self.box_head(feats1.flatten(1, -1))

        frame_features0 = self.frames_linear(self.flatten(self.global_pool(raw_feats0)))
        frame_features0 = torch.index_select(frame_features0, 0, inputs["boxes0"][:, 4].int())

        frame_features1 = self.frames_linear(self.flatten(self.global_pool(raw_feats1)))
        frame_features1 = torch.index_select(frame_features1, 0, inputs["boxes1"][:, 4].int())

        visual_feats = torch.cat([feats0, frame_features0, feats1, frame_features1], 1)

        track_feats = self.mlp(inputs["features"])
        logits = self.fc(torch.cat([visual_feats, track_feats], 1))
        return logits
