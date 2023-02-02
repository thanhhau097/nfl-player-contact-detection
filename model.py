import timm
import torch
from torch import nn
import torch.nn.functional as F

# from backbone import Resnet, Convnext, Efficientnet

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class Model(nn.Module):
    # def __init__(self, model_name="resnet50", in_chans=13):
    def __init__(self, model_name="resnet50"):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            # model_name, pretrained=True, in_chans=in_chans, drop_path_rate=0.2
            model_name, pretrained=True, drop_path_rate=0.2
        )
        self.backbone.reset_classifier(0, "")
        num_features = self.backbone.num_features
        # if self.model_name.startswith('resnet50'):
        #     self.backbone = Resnet(model_name)
        # elif self.model_name.startswith('convnext'):
        #     self.backbone = Convnext(model_name)
        # else:
        #     self.backbone = Efficientnet(model_name)
        # num_features = self.backbone.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention_pool = AttentionPool2d(
            spacial_dim = 256//32, 
            embed_dim = num_features,
            num_heads = 2
        )
        self.flatten = nn.Flatten()

        self.model_name = model_name
        if self.model_name.startswith('resnet'):
            self.frames_linear = nn.Sequential(
                nn.Linear(self.backbone.num_features * 2, self.backbone.num_features), 
                nn.LayerNorm(self.backbone.num_features), 
                nn.ReLU(), 
                nn.Dropout(0.2)
            )
            # self.frames_linear = nn.Sequential(nn.Linear(num_features, num_features), nn.LayerNorm(self.backbone.num_features), nn.ReLU(), nn.Dropout(0.2))
        else:
            if self.model_name.startswith('convnext'):
                self.frames_linear = nn.Sequential(nn.Linear(num_features * 2, num_features), nn.ReLU())
            else:
                self.frames_linear = nn.Sequential(nn.Linear(num_features * 4, num_features), nn.ReLU())
            self.aggregater = nn.Sequential(
                                nn.Conv3d(num_features, num_features, kernel_size=(3,1,1), stride=(2, 1, 1)), 
                                nn.MaxPool3d(kernel_size=(2,1,1), stride=(2, 1, 1))
                            )
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
        self.fc = nn.Linear(64 + num_features, 1)

    def forward(self, images, features, contact_ids, labels=None):
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

        # images = self.frames_linear(self.flatten(self.global_pool(images)))
        feat_1 = self.flatten(self.attention_pool(images))
        feat_2 = self.flatten(self.global_pool(images))
        images = self.frames_linear(torch.cat([feat_1, feat_2], -1))
        features = self.mlp(features)
        y = self.fc(torch.cat([images, features], dim=1))
        # _b, _f = images.shape
        # images = images.reshape(_b, _f//4, 4).mean(dim=-1)
        # return y, torch.cat([images, features], dim=1)
        return y
