"""DenseNet-121 from scratch in PyTorch, for 2D images or 3D volumes.

Follows Huang et al. (2017), "Densely Connected Convolutional Networks".
The argument names match `monai.networks.nets.DenseNet121`, which is what the LUNA16 training scripts use. With
`spatial_dims=3, in_channels=1, out_channels=2` this implementation has exactly 11,244,674 parameters, the same as the
MONAI model that produced the trained checkpoints (checked in tests/test_models.py).

    Stem conv -> [Dense block -> Transition] x 3 -> Dense block -> BN-ReLU -> global average pool -> Linear

The 7.98 M parameter figure often quoted for DenseNet-121 is the 2D / 1000-class ImageNet model; the 3D version used here
has more parameters because every convolution kernel gains a dimension.
"""
from typing import Sequence

import torch
import torch.nn as nn


def _layers(spatial_dims: int):
    if spatial_dims not in (2, 3):
        raise ValueError("spatial_dims must be 2 or 3")
    return (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d) if spatial_dims == 2 else \
           (nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d, nn.AvgPool3d, nn.AdaptiveAvgPool3d)


class BottleneckLayer(nn.Module):
    """BN -> ReLU -> Conv 1x1 (to bn_size*k channels) -> BN -> ReLU -> Conv 3x3 (to k channels), output concatenated to the input."""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int, spatial_dims: int, dropout: float = 0.0):
        super().__init__()
        Conv, Norm, *_ = _layers(spatial_dims)
        inter = bn_size * growth_rate
        self.norm1, self.conv1 = Norm(in_channels), Conv(in_channels, inter, kernel_size=1, bias=False)
        self.norm2, self.conv2 = Norm(inter), Conv(inter, growth_rate, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.act(self.norm1(x)))
        out = self.conv2(self.act(self.norm2(out)))
        return torch.cat([x, self.dropout(out)], dim=1)


class DenseBlock(nn.Sequential):
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int, bn_size: int, spatial_dims: int, dropout: float = 0.0):
        super().__init__(*[
            BottleneckLayer(in_channels + i * growth_rate, growth_rate, bn_size, spatial_dims, dropout)
            for i in range(num_layers)
        ])


class TransitionLayer(nn.Sequential):
    """BN -> ReLU -> Conv 1x1 (compress channels) -> AvgPool 2."""

    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int):
        Conv, Norm, _, AvgPool, _ = _layers(spatial_dims)
        super().__init__(Norm(in_channels), nn.ReLU(inplace=True),
                         Conv(in_channels, out_channels, kernel_size=1, bias=False), AvgPool(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    def __init__(self, spatial_dims: int = 3, in_channels: int = 1, out_channels: int = 2,
                 init_features: int = 64, growth_rate: int = 32, block_config: Sequence[int] = (6, 12, 24, 16),
                 bn_size: int = 4, compression: float = 0.5, dropout: float = 0.0):
        super().__init__()
        Conv, Norm, MaxPool, _, AdaptivePool = _layers(spatial_dims)
        self.features = nn.Sequential(
            Conv(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            Norm(init_features), nn.ReLU(inplace=True), MaxPool(kernel_size=3, stride=2, padding=1))

        channels = init_features
        for i, num_layers in enumerate(block_config):
            self.features.add_module(f"denseblock{i + 1}", DenseBlock(channels, num_layers, growth_rate, bn_size, spatial_dims, dropout))
            channels += num_layers * growth_rate
            if i != len(block_config) - 1:
                compressed = int(channels * compression)
                self.features.add_module(f"transition{i + 1}", TransitionLayer(channels, compressed, spatial_dims))
                channels = compressed

        self.features.add_module("norm5", Norm(channels))
        self.head = nn.Sequential(nn.ReLU(inplace=True), AdaptivePool(1), nn.Flatten(), nn.Linear(channels, out_channels))
        self.feature_channels = channels
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for dims, shape in ((2, (2, 1, 64, 64)), (3, (2, 1, 64, 64, 64))):
        model = DenseNet121(spatial_dims=dims)
        print(f"{dims}D: {model.count_parameters():,} parameters, output {tuple(model(torch.randn(*shape)).shape)}")
