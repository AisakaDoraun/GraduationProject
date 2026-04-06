import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """
    双卷积模块：Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU

    这是U-Net架构中的基本构建块，用于特征提取。
    包含两个连续的卷积层，每个卷积层后接批归一化和ReLU激活函数。

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        mid_channels (Optional[int]): 中间通道数，如果未指定则使用out_channels
    """

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        # 如果未指定中间通道数，则使用输出通道数
        if not mid_channels:
            mid_channels = out_channels

        # 定义双卷积序列
        self.double_conv = nn.Sequential(
            # 第一层卷积：3x3卷积，保持空间尺寸不变
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化：加速训练，提高稳定性
            nn.BatchNorm2d(mid_channels),
            # ReLU激活函数：引入非线性，inplace=True节省内存
            nn.ReLU(inplace=True),
            # 第二层卷积：3x3卷积
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化
            nn.BatchNorm2d(out_channels),
            # ReLU激活函数
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, height, width)
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样模块：最大池化 -> 双卷积

    用于U-Net编码器部分，降低特征图分辨率的同时增加通道数。

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 定义下采样序列：2x2最大池化 + 双卷积
        self.maxpool_conv = nn.Sequential(
            # 2x2最大池化：将特征图尺寸减半
            nn.MaxPool2d(2),
            # 双卷积模块：提取特征
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, height/2, width/2)
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样模块：上采样 -> 特征拼接 -> 双卷积

    用于U-Net解码器部分，恢复特征图分辨率的同时减少通道数。
    支持双线性插值和转置卷积两种上采样方式。

    参数:
        in_channels (int): 输入通道数（来自编码器的特征图）
        out_channels (int): 输出通道数
        bilinear (bool): 是否使用双线性插值上采样，True为双线性插值，False为转置卷积
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            # 双线性插值上采样：计算效率高，但可能丢失细节
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # 双卷积模块：输入通道数为in_channels（拼接后），中间通道数为in_channels//2
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 转置卷积上采样：可学习参数，可能恢复更多细节
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            # 双卷积模块
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x1 (torch.Tensor): 来自解码器的特征图，需要上采样
            x2 (torch.Tensor): 来自编码器的对应特征图，用于跳跃连接

        返回:
            torch.Tensor: 输出张量
        """
        # 上采样x1
        x1 = self.up(x1)

        # 计算尺寸差异（由于池化/上采样可能产生尺寸不匹配）
        diff_y = x2.size()[2] - x1.size()[2]  # 高度差异
        diff_x = x2.size()[3] - x1.size()[3]  # 宽度差异

        # 填充x1使其与x2尺寸匹配
        # 填充顺序：[左, 右, 上, 下]
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        # 沿通道维度拼接特征图（跳跃连接）
        x = torch.cat([x2, x1], dim=1)

        # 通过双卷积模块
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出卷积层：1x1卷积生成最终分割图

    将特征图映射到分割类别空间，不改变空间分辨率。

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数（等于分割类别数）
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 1x1卷积：将特征通道数映射到类别数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)

        返回:
            torch.Tensor: 输出分割图，形状为 (batch_size, out_channels, height, width)
        """
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net图像分割模型

    经典的编码器-解码器架构，具有跳跃连接，用于像素级图像分割。
    编码器提取多尺度特征，解码器恢复空间分辨率并生成分割图。

    参数:
        in_channels (int): 输入图像通道数（3表示RGB图像）
        num_classes (int): 分割类别数
        bilinear (bool): 是否使用双线性插值上采样，True为双线性插值，False为转置卷积
        base_channels (int): 基础特征通道数，控制模型容量
        dropout (float): Dropout概率，用于正则化防止过拟合
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
        # 保存模型参数
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 如果使用双线性插值，上采样后通道数减半
        factor = 2 if bilinear else 1

        # 编码器部分（下采样路径）
        # 输入卷积：提取初始特征
        self.inc = DoubleConv(in_channels, base_channels)
        # 下采样层1：64 -> 128通道，分辨率减半
        self.down1 = Down(base_channels, base_channels * 2)
        # 下采样层2：128 -> 256通道，分辨率减半
        self.down2 = Down(base_channels * 2, base_channels * 4)
        # 下采样层3：256 -> 512通道，分辨率减半
        self.down3 = Down(base_channels * 4, base_channels * 8)
        # 下采样层4：512 -> 1024//factor通道，分辨率减半
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # 解码器部分（上采样路径）
        # 上采样层1：1024 -> 512//factor通道，分辨率加倍
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        # 上采样层2：512 -> 256//factor通道，分辨率加倍
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        # 上采样层3：256 -> 128//factor通道，分辨率加倍
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        # 上采样层4：128 -> 64通道，分辨率加倍
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        # 输出层：64 -> num_classes通道，生成分割图
        self.outc = OutConv(base_channels, num_classes)

        # Dropout层：如果dropout>0则启用，用于正则化
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x (torch.Tensor): 输入图像，形状为 (batch_size, in_channels, height, width)

        返回:
            torch.Tensor: 分割logits，形状为 (batch_size, num_classes, height, width)
        """
        # 编码器路径
        x1 = self.inc(x)  # 初始特征提取
        x2 = self.down1(x1)  # 第一次下采样
        x3 = self.down2(x2)  # 第二次下采样
        x4 = self.down3(x3)  # 第三次下采样
        x5 = self.down4(x4)  # 第四次下采样（瓶颈层）

        # 在瓶颈层应用Dropout（如果启用）
        if self.dropout:
            x5 = self.dropout(x5)

        # 解码器路径（带跳跃连接）
        x = self.up1(x5, x4)  # 第一次上采样，与x4拼接
        x = self.up2(x, x3)  # 第二次上采样，与x3拼接
        x = self.up3(x, x2)  # 第三次上采样，与x2拼接
        x = self.up4(x, x1)  # 第四次上采样，与x1拼接

        # 在最终特征上应用Dropout（如果启用）
        if self.dropout:
            x = self.dropout(x)

        # 生成最终分割图
        logits = self.outc(x)
        return logits


def build_unet(
    in_channels: int = 3, num_classes: int = 2, pretrained: bool = False, **kwargs
) -> UNet:
    """
    构建U-Net模型

    参数:
        in_channels (int): 输入图像通道数
        num_classes (int): 分割类别数
        pretrained (bool): 是否加载预训练权重（暂不支持）
        **kwargs: 传递给UNet构造函数的额外参数

    返回:
        UNet: 构建好的U-Net模型实例
    """
    model = UNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    """
    模型测试代码
    用于验证模型构建是否正确，计算参数量，测试前向传播
    """
    # 构建模型
    model = build_unet(in_channels=3, num_classes=2, bilinear=True)

    # 打印模型结构
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)  # 批量大小2，3通道，256x256图像
    y = model(x)
    print(f"\n输入形状:  {x.shape}")
    print(f"输出形状: {y.shape}")
