from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights, MobileNet_V2_Weights, ResNet18_Weights

from config import CLASS_NAMES
from .attention import ChannelAttention, SpatialAttention
from .oect import OECTFrontEnd


SUPPORTED_BACKBONES = ("resnet18", "mobilenet_v2", "densenet121", "vit", "swin")
SUPPORTED_INITIAL_CONV_MODES = ("fixed", "learnable", "none")
SUPPORTED_FIXED_FRONT_END_KINDS = ("generic", "oect")


class CustomCNN(nn.Module):
    def __init__(
        self,
        conv_kernel=None,
        backbone_name="resnet18",
        pretrained=False,
        initial_conv_mode=None,
        fixed_front_end_kind="generic",
        oect_gate_voltage=None,
        oect_response_source=None,
    ):
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose one of: {', '.join(SUPPORTED_BACKBONES)}"
            )

        if initial_conv_mode is None:
            initial_conv_mode = "fixed" if conv_kernel is not None else "none"
        if initial_conv_mode not in SUPPORTED_INITIAL_CONV_MODES:
            raise ValueError(
                f"Unsupported initial_conv_mode '{initial_conv_mode}'. "
                f"Choose one of: {', '.join(SUPPORTED_INITIAL_CONV_MODES)}"
            )
        if fixed_front_end_kind not in SUPPORTED_FIXED_FRONT_END_KINDS:
            raise ValueError(
                f"Unsupported fixed_front_end_kind '{fixed_front_end_kind}'. "
                f"Choose one of: {', '.join(SUPPORTED_FIXED_FRONT_END_KINDS)}"
            )
        if initial_conv_mode != "fixed" and fixed_front_end_kind != "generic":
            raise ValueError("fixed_front_end_kind is only configurable for fixed front ends")

        self.backbone_name = backbone_name
        self.initial_conv_mode = initial_conv_mode
        self.fixed_front_end_kind = fixed_front_end_kind
        self.front_end_kind = (
            fixed_front_end_kind if initial_conv_mode == "fixed" else initial_conv_mode
        )
        self.initial_conv = self._build_initial_conv(
            initial_conv_mode,
            conv_kernel,
            fixed_front_end_kind,
            oect_gate_voltage,
            oect_response_source,
        )
        self._init_backbone(backbone_name, pretrained)
        self._init_attention()
        self.classifier = self._build_classifier()

    @staticmethod
    def _build_initial_conv(
        mode,
        conv_kernel,
        fixed_front_end_kind,
        oect_gate_voltage,
        oect_response_source,
    ):
        if mode == "none":
            if conv_kernel is not None:
                raise ValueError("conv_kernel must be omitted when initial_conv_mode='none'")
            return None

        if mode == "fixed":
            if conv_kernel is None:
                raise ValueError("A 3x3 conv_kernel is required for initial_conv_mode='fixed'")
            if fixed_front_end_kind == "oect":
                return OECTFrontEnd(
                    response_kernel=conv_kernel,
                    gate_voltage=oect_gate_voltage,
                    response_source=oect_response_source,
                )

            kernel_tensor = torch.as_tensor(conv_kernel, dtype=torch.float32)
            if kernel_tensor.shape != (3, 3):
                raise ValueError(
                    "Fixed convolution kernel must have shape (3, 3), "
                    f"got {tuple(kernel_tensor.shape)}"
                )
            if not torch.isfinite(kernel_tensor).all():
                raise ValueError("Fixed convolution kernel contains non-finite values")
            layer = nn.Conv2d(1, 1, kernel_size=3, stride=3, bias=False)
            with torch.no_grad():
                layer.weight.copy_(kernel_tensor.reshape(1, 1, 3, 3))
            layer.requires_grad_(False)
            return layer

        if conv_kernel is not None:
            raise ValueError("conv_kernel is only valid for initial_conv_mode='fixed'")
        return nn.Conv2d(1, 1, kernel_size=3, stride=3, bias=False)

    def _init_backbone(self, name, pretrained):
        if name == "resnet18":
            model = models.resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.maxpool = nn.Identity()
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            self.feature_dim = 512
        elif name == "mobilenet_v2":
            model = models.mobilenet_v2(
                weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            )
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone = model.features
            self.feature_dim = 1280
        elif name == "densenet121":
            model = models.densenet121(
                weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            )
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = model.features
            self.feature_dim = 1024
        elif name == "vit":
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
            )
            self.feature_dim = 768
        else:  # swin
            self.backbone = timm.create_model(
                "swin_base_patch4_window7_224",
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
            )
            self.feature_dim = 1024

    def _init_attention(self):
        if self.backbone_name in ("vit", "swin"):
            self.channel_att = None
            self.spatial_att = None
        else:
            self.channel_att = ChannelAttention(self.feature_dim)
            self.spatial_att = SpatialAttention()

    def _build_classifier(self):
        if self.backbone_name in ("vit", "swin"):
            return nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, 512),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(CLASS_NAMES)),
            )
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(CLASS_NAMES)),
        )

    def forward(self, inputs):
        if self.initial_conv is not None:
            inputs = self.initial_conv(inputs)

        if self.backbone_name in ("vit", "swin"):
            inputs = F.interpolate(
                inputs,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
        features = self.backbone(inputs)

        if self.channel_att is not None:
            features = features * self.channel_att(features)
        if self.spatial_att is not None:
            features = features * self.spatial_att(features)
        return self.classifier(features)


def load_model_checkpoint(model, checkpoint_path, map_location="cpu", strict=True):
    """Load a metadata checkpoint or a legacy plain state_dict into a model."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        payload = torch.load(path, map_location=map_location)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state_dict = dict(payload)
        payload = {"model_state_dict": state_dict, "legacy_checkpoint": True}
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {path}; expected a metadata checkpoint or state_dict"
        )

    model_keys = model.state_dict().keys()
    adapted_keys = []
    if payload.get("legacy_checkpoint") and "initial_conv.bias" in state_dict and "initial_conv.bias" not in model_keys:
        state_dict = dict(state_dict)
        state_dict.pop("initial_conv.bias")
        adapted_keys.append("initial_conv.bias")
    if adapted_keys:
        payload["adapted_legacy_keys"] = adapted_keys

    model.load_state_dict(state_dict, strict=strict)
    return payload
