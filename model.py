import timm
import torch
from torch import nn
from torchvision.ops import RoIAlign


class Model(nn.Module):
    def __init__(self, model_name="resnet50", in_chans=13):
        super(Model, self).__init__()
        self.in_chans = in_chans
        self.backbone = timm.create_model(
            model_name, pretrained=True, in_chans=1, drop_path_rate=0.2, num_classes=1024
        )
        # self.global_pool = self.backbone.global_pool
        # try:
        #     self.backbone_fc = self.backbone.fc
        # except:
        #     self.backbone_fc = self.backbone.classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.frames_linear = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU()
        )

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
            nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 + 512 * 2 + 512 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, inputs):
        # still 1 channel: TODO: update to use 3 channels
        # Endzone
        # b, c, h, w = inputs["images0"].shape
        # raw_feats0 = self.backbone(inputs["images0"].reshape(b * c, 1, w, h))

        # box_b, box_c, box_len = inputs["boxes0"].shape
        # feats0 = self.box_roi_pool(raw_feats0, inputs["boxes0"].reshape(box_b * box_c, box_len))
        # feats0 = self.box_head(feats0.flatten(1, -1))
        # # Sideline
        # raw_feats1 = self.backbone(inputs["images1"].reshape(b * c, 1, w, h))
        # feats1 = self.box_roi_pool(raw_feats1, inputs["boxes1"].reshape(box_b * box_c, box_len))
        # feats1 = self.box_head(feats1.flatten(1, -1))

        # frame_features0 = self.frames_linear(self.flatten(self.global_pool(raw_feats0)))
        # frame_features0 = torch.index_select(frame_features0, 0, inputs["boxes0"].reshape(box_b * box_c, box_len)[:, 4].int())
        
        # frame_features1 = self.frames_linear(self.flatten(self.global_pool(raw_feats1)))
        # frame_features1 = torch.index_select(frame_features1, 0, inputs["boxes1"].reshape(box_b * box_c, box_len)[:, 4].int())
    
        # visual_feats = torch.cat([feats0, frame_features0, feats1, frame_features1], 1)
        # visual_feats = visual_feats.reshape(box_b, box_c, 2048).mean(1)

        all_visual_features = []
        for i in range(self.in_chans):
            # Endzone
            raw_feats0 = self.backbone(inputs["images0"][:, i * 3: i * 3 + 1])
            feats0 = self.box_roi_pool(raw_feats0, inputs["boxes0"][:, i])
            feats0 = self.box_head(feats0.flatten(1, -1))
            # Sideline
            raw_feats1 = self.backbone(inputs["images1"][:, i * 3: i * 3 + 1])
            feats1 = self.box_roi_pool(raw_feats1, inputs["boxes1"][:, i])
            feats1 = self.box_head(feats1.flatten(1, -1))

            frame_features0 = self.frames_linear(self.flatten(self.global_pool(raw_feats0)))
            frame_features0 = torch.index_select(frame_features0, 0, inputs["boxes0"][:, i, 4].int())
            
            frame_features1 = self.frames_linear(self.flatten(self.global_pool(raw_feats1)))
            frame_features1 = torch.index_select(frame_features1, 0, inputs["boxes1"][:, i, 4].int())
        
            visual_feats = torch.cat([feats0, frame_features0, feats1, frame_features1], 1)

            all_visual_features.append(visual_feats)

        visual_feats = torch.stack(all_visual_features, -1).mean(2)

        track_feats = self.mlp(inputs["features"])
        logits = self.fc(torch.cat([visual_feats, track_feats], 1))
        return logits
