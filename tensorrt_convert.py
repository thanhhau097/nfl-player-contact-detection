import timm
import torch
from torch import nn
from torch2trt import torch2trt

CFG = {
    'model': 'tf_efficientnetv2_b2',
#     'model': 'tf_efficientnetv2_s_in21ft1k',
#     'model': 'convnext_base',
#     'model': 'resnet50',
}

class Model_conv3d(nn.Module):
    def __init__(self):
        super(Model_conv3d, self).__init__()
        self.backbone = timm.create_model(CFG['model'], pretrained=False, drop_path_rate=0.)
        self.backbone.reset_classifier(0, "")
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.frames_linear = nn.Sequential(nn.Linear(self.backbone.num_features * 2, self.backbone.num_features), nn.ReLU())
        self.aggregater = nn.Sequential(
                            nn.Conv3d(self.backbone.num_features, self.backbone.num_features, kernel_size=(3,1,1), stride=(2, 1, 1)), 
                            nn.MaxPool3d(kernel_size=(2,1,1), stride=(2, 1, 1))
                        )
        
        self.mlp = nn.Sequential(
            nn.Linear(18, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(64 + self.backbone.num_features, 1)

    def forward(self, images, features):
        print("input")
        print(images.shape)
        print(features.shape)
        b, c, h, w = images.shape
        
        # images = images.reshape(b * 2, c // 2, h, w)
        # images = self.backbone(images).reshape(b, -1)

        images = images.reshape(b, c // 3, 3, h, w).flatten(0, 1)
        images = self.backbone(images)
        _, _c, _h, _w = images.shape
        images = images.reshape(b, _c, c // 3, _h, _w)
        images = self.aggregater(images)
        images = self.frames_linear(self.flatten(self.global_pool(images)))

        features = self.mlp(features)
        y = self.fc(torch.cat([images, features], dim=1))
        print("output")
        print(y.shape)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model..")
model = Model_conv3d().to(device)
ckpt = torch.load('upload/b2_block_conv3d_692.pth')
model.load_state_dict(ckpt)
model.eval()

print("Converting model..")
model_trt = torch2trt(
                model, 
                [torch.zeros((4, 30, 256, 256)).cuda(), torch.zeros((4, 18)).cuda()], 
                min_shapes=[(1, 30, 256, 256), (1, 18)], 
                max_shapes=[(4, 30, 256, 256), (4, 18)]
            )   
print("Saving model..")
torch.save(model_trt.state_dict(), "trt_b2.pth")