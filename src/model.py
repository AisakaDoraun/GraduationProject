import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """DoubleConv: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU"""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution to produce final segmentation map"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net image segmentation model.

    Args:
        in_channels: Input image channels (3 for RGB).
        num_classes: Number of segmentation classes.
        bilinear: Use bilinear upsampling instead of transposed conv.
        base_channels: Number of base feature channels.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        bilinear: bool = True,
        base_channels: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.dropout:
            x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.dropout:
            x = self.dropout(x)

        logits = self.outc(x)
        return logits


def build_unet(
    in_channels: int = 3, num_classes: int = 2, pretrained: bool = False, **kwargs
) -> UNet:
    """
    Build a U-Net model.

    Args:
        in_channels: Input image channels.
        num_classes: Number of segmentation classes.
        pretrained: Whether to load pretrained weights (not yet supported).
        **kwargs: Additional arguments passed to UNet.
    """
    model = UNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    model = build_unet(in_channels=3, num_classes=2, bilinear=True)

    print(model)
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
