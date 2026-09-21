"""Explicit baseline names; model creation never downloads weights."""
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .io import ROOT


class ConvBlock(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1),
                         nn.BatchNorm2d(outgoing), nn.ReLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1),
                         nn.BatchNorm2d(outgoing), nn.ReLU())


class UNet(nn.Module):
    """Conventional U-Net encoder/decoder, logits, configurable base width."""
    def __init__(self, width=32):
        super().__init__()
        channels = [width * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList([ConvBlock(3, channels[0])] +
                                     [ConvBlock(a, b) for a, b in zip(channels, channels[1:])])
        self.decoders = nn.ModuleList([ConvBlock(channels[i] + channels[i-1], channels[i-1])
                                      for i in range(4, 0, -1)])
        self.head = nn.Conv2d(width, 2, 1)

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.encoders):
            x = layer(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for i, layer in enumerate(self.decoders):
            skip = skips[-2-i]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat([x, skip], 1))
        return self.head(x)


class TorchvisionAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def segmentation_model(config):
    name = config["name"]
    if name == "unet":
        return UNet(config.get("width", 32))
    if name in {"deeplab_mobilenet", "deeplab_xception", "segnet"}:
        source = str(ROOT / "segment")
        if source not in sys.path:
            sys.path.insert(0, source)
        if name == "segnet":
            from nets.segnet import SegNET
            return SegNET(input_channels=3, num_classes=2)
        from nets.deeplabv3_plus import DeepLab
        return DeepLab(num_classes=2, backbone=name.split("_")[1], pretrained=False,
                       downsample_factor=config.get("output_stride", 16))
    if name == "lraspp_mobilenet":
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        return TorchvisionAdapter(lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=2))
    if name == "segformer_b0":
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError("Optional SegFormer requires transformers; install on GPU host") from exc
        class SegformerAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=2))
            def forward(self, x):
                logits = self.model(pixel_values=x).logits
                return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return SegformerAdapter()
    raise ValueError("Unknown segmentation model: {}".format(name))


class GeometryCNN(nn.Module):
    """Boundary/area branches with nonlinearities and separate input dimensions."""
    def __init__(self, n_area, n_boundary, hidden=32):
        super().__init__()
        if n_area < 1 or n_boundary < 2 or n_boundary % 2:
            raise ValueError("dual_cnn needs area and equally sized upper/lower boundary features")
        self.n_area = n_area
        self.area = nn.Sequential(nn.Linear(n_area, hidden), nn.ReLU(), nn.Dropout(.1))
        self.boundary = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(4*n_boundary, hidden), nn.ReLU(), nn.Dropout(.1))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.head(torch.cat([self.area(x[:, :self.n_area]),
                                    self.boundary(x[:, self.n_area:].reshape(len(x), 2, -1))], 1))
