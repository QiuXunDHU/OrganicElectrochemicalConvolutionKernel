import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights, DenseNet121_Weights
import timm
from config import CLASS_NAMES
from .attention import ChannelAttention, SpatialAttention


class CustomCNN(nn.Module):
    def __init__(self, conv_kernel=None, backbone_name='resnet18', pretrained=True):
        super().__init__()
        self.initial_conv = None
        self.backbone_name = backbone_name

        if conv_kernel is not None:
            kernel_tensor = torch.tensor(conv_kernel, dtype=torch.float32)
            self.initial_conv = nn.Conv2d(1, 1, kernel_size=3, stride=3)
            self.initial_conv.weight.data = kernel_tensor.view(1, 1, 3, 3)
            self.initial_conv.requires_grad_(False)

        self._init_backbone(backbone_name, pretrained)
        self._init_attention()
        self.classifier = self._build_classifier()

    def _init_backbone(self, name, pretrained):
        if name == 'resnet18':
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.maxpool = nn.Identity()
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            self.feature_dim = 512

        elif name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone = model.features
            self.feature_dim = 1280

        elif name == 'densenet121':
            model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = model.features
            self.feature_dim = 1024

        elif name == 'vit':
            self.backbone = timm.create_model('vit_base_patch16_224',
                                              pretrained=pretrained,
                                              in_chans=1,
                                              num_classes=0)
            self.feature_dim = 768
            self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

        elif name == 'swin':
            self.backbone = timm.create_model('swin_base_patch4_window7_224',
                                              pretrained=pretrained,
                                              in_chans=1,
                                              num_classes=0)
            self.feature_dim = 1024
            self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def _init_attention(self):
        if 'vit' in self.backbone_name or 'swin' in self.backbone_name:
            self.channel_att = None
            self.spatial_att = None
        else:
            self.channel_att = ChannelAttention(self.feature_dim)
            self.spatial_att = SpatialAttention()

    def _build_classifier(self):
        if 'vit' in self.backbone_name or 'swin' in self.backbone_name:
            return nn.Sequential(
                nn.LayerNorm(self.feature_dim),  # 新增层归一化
                nn.Linear(self.feature_dim, 512),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(CLASS_NAMES))
            )
        else:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(CLASS_NAMES))
            )

    def forward(self, x):
        if self.initial_conv is not None:
            x = self.initial_conv(x)

        if 'vit' in self.backbone_name or 'swin' in self.backbone_name:
            x = F.interpolate(x, size=224, mode='bilinear')
            x = self.backbone(x)  # 输出形状为 (batch_size, feature_dim)
        else:
            x = self.backbone(x)

        if self.channel_att is not None:
            x = x * self.channel_att(x)
        if self.spatial_att is not None:
            x = x * self.spatial_att(x)

        return self.classifier(x)
