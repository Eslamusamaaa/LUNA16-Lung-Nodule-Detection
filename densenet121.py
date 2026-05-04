"""
DenseNet-121 — Generic Implementation from Scratch
====================================================
PyTorch implementation following the original paper:
  "Densely Connected Convolutional Networks" (Huang et al., 2017)

Architecture:
  Stem Conv → [Dense Block → Transition] × 3 → Dense Block → Head

Block config : [6, 12, 24, 16]
Growth rate  : k = 32
Compression  : θ = 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
#  Bottleneck Layer
# ──────────────────────────────────────────────────────────────

class BottleneckLayer(nn.Module):
    """
    One dense layer (bottleneck variant):
        BN → ReLU → Conv 1×1 (→ 4k channels)
        BN → ReLU → Conv 3×3 (→ k  channels)

    Output is concatenated with the input (dense connection).
    """

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        inter = 4 * growth_rate

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter,
                               kernel_size=1, bias=False)

        self.bn2   = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth_rate,
                               kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)   # dense skip


# ──────────────────────────────────────────────────────────────
#  Dense Block
# ──────────────────────────────────────────────────────────────

class DenseBlock(nn.Module):
    """
    Stack of `num_layers` bottleneck layers.
    Each layer receives the concatenated outputs of all previous layers.
    """

    def __init__(self, in_channels: int, num_layers: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(num_layers):
            self.layers.append(BottleneckLayer(ch, growth_rate))
            ch += growth_rate
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ──────────────────────────────────────────────────────────────
#  Transition Layer
# ──────────────────────────────────────────────────────────────

class TransitionLayer(nn.Module):
    """
    BN → ReLU → Conv 1×1 (compress channels by θ) → AvgPool 2×2
    """

    def __init__(self, in_channels: int, compression: float = 0.5):
        super().__init__()
        out = int(in_channels * compression)
        self.bn   = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.bn(x), inplace=True))
        return self.pool(x)


# ──────────────────────────────────────────────────────────────
#  DenseNet-121
# ──────────────────────────────────────────────────────────────

class DenseNet121(nn.Module):
    """
    Generic DenseNet-121.

    Args:
        in_channels  : number of input image channels (default 3 for RGB)
        num_classes  : number of output classes (default 1000)
        growth_rate  : k, feature maps added per layer (default 32)
        compression  : θ, channel reduction in transitions (default 0.5)
        drop_rate    : dropout probability before classifier (default 0.0)
    """

    BLOCK_CONFIG = (6, 12, 24, 16)

    def __init__(self,
                 in_channels: int  = 3,
                 num_classes: int  = 1000,
                 growth_rate: int  = 32,
                 compression: float = 0.5,
                 drop_rate: float  = 0.0):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────
        stem_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── Dense Blocks + Transitions ────────────────────────
        self.blocks      = nn.ModuleList()
        self.transitions = nn.ModuleList()
        ch = stem_ch

        for i, num_layers in enumerate(self.BLOCK_CONFIG):
            block = DenseBlock(ch, num_layers, growth_rate)
            self.blocks.append(block)
            ch = block.out_channels

            if i < len(self.BLOCK_CONFIG) - 1:          # no transition after last block
                trans = TransitionLayer(ch, compression)
                self.transitions.append(trans)
                ch = trans.out_channels

        # ── Head ──────────────────────────────────────────────
        self.bn_final  = nn.BatchNorm2d(ch)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout   = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(ch, num_classes)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            logits : (B, num_classes)
        """
        x = self.stem(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = F.relu(self.bn_final(x), inplace=True)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = DenseNet121(in_channels=3, num_classes=1000)
    x     = torch.randn(2, 3, 224, 224)
    out   = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Params: {model.count_parameters():,}")
