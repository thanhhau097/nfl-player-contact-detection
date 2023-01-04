import timm
import torch
from torch import nn
from torch.nn import functional as F
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


class ResidualLSTM(nn.Module):
    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM = nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1 = nn.Linear(d_model * 2, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        res = x
        x, _ = self.LSTM(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = res + x
        return x


class SeqModel(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        embed_dim=128,
        pos_encode="LSTM",
        nlayers=2,
        rnnlayers=3,
        dropout=0.1,
        nheads=8,
    ):
        super(SeqModel, self).__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim

        if pos_encode == "LSTM":
            self.pos_encoder = nn.ModuleList([ResidualLSTM(embed_dim) for i in range(rnnlayers)])

        self.pos_encoder_dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(in_features, embed_dim)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.transformer_encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim * 4, dropout)
                for i in range(nlayers)
            ]
        )
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding=1) for i in range(nlayers)]
        )
        self.layer_norm_layers = nn.ModuleList(nn.LayerNorm(embed_dim) for i in range(nlayers))
        # self.layer_norm_layers2 = nn.ModuleList(nn.LayerNorm(embed_dim) for i in range(nlayers))
        # self.deconv_layers = nn.ModuleList(
        #     [
        #         nn.ConvTranspose1d(embed_dim, embed_dim, 5, stride=1, padding=0)
        #         for i in range(nlayers)
        #     ]
        # )
        self.nheads = nheads
        self.pred = nn.Linear(embed_dim, out_features)

    def forward(self, features):
        x = self.embedding(features)
        x = x.permute(1, 0, 2)
        for lstm in self.pos_encoder:
            lstm.LSTM.flatten_parameters()
            x = lstm(x)
        x = self.pos_encoder_dropout(x)
        x = self.layer_normal(x)

        # for layer in self.transformer_encoder:
        #     x = layer(x)

        for conv, transformer_layer, layer_norm1 in zip(
            self.conv_layers,
            self.transformer_encoder,
            self.layer_norm_layers,
            # self.layer_norm_layers2,
            # self.deconv_layers,
        ):
            res = x
            x = F.relu(conv(x.permute(1, 2, 0)).permute(2, 0, 1))
            x = layer_norm1(x)
            x = transformer_layer(x)
            # x = F.relu(deconv(x.permute(1, 2, 0)).permute(2, 0, 1))
            # x = layer_norm2(x)
            x = res + x

        x = x.permute(1, 0, 2)

        output = self.pred(x)

        return output.squeeze(-1)
