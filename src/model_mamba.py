import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# 导入基础组件（复用 model.py 中的逻辑或重写以适配 Mamba）
class DoubleConv(nn.Module):
    """
    双卷积模块：(Conv2d -> BatchNorm -> ReLU) * 2

    U-Net架构中的基本构建块，用于特征提取。
    与model.py中的DoubleConv类似，但为Mamba U-Net适配。

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 第一层卷积：3x3卷积，保持空间尺寸
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化：加速训练，提高稳定性
            nn.BatchNorm2d(out_channels),
            # ReLU激活函数：引入非线性
            nn.ReLU(inplace=True),
            # 第二层卷积：3x3卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化
            nn.BatchNorm2d(out_channels),
            # ReLU激活函数
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        return self.double_conv(x)


class MambaBlock(nn.Module):
    """
    标准 1D Mamba 块

    Mamba是一种选择性状态空间模型（SSM），具有线性时间复杂度和高效的长序列建模能力。
    这是Mamba块的简化实现，生产环境建议使用官方mamba_ssm库。

    参数:
        dim (int): 输入/输出维度
        d_state (int): 状态维度，默认16
        d_conv (int): 卷积核大小，默认4
        expand (int): 扩展因子，默认2
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim  # 输入维度
        self.d_inner = int(expand * dim)  # 内部维度（扩展后）

        # 输入投影层：将输入投影到内部维度的两倍（用于门控）
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # 1D深度卷积：用于局部特征提取
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )

        # 状态空间参数投影层
        self.x_proj = nn.Linear(
            self.d_inner, 16 + d_state * 2, bias=False
        )  # dt_rank=16

        # 时间步参数投影层
        self.dt_proj = nn.Linear(16, self.d_inner, bias=True)

        # 状态转移矩阵A的对数参数（可学习）
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1))
        )

        # 跳跃连接参数D（可学习）
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影层：将内部维度映射回原始维度
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # 激活函数：SiLU（Swish）
        self.act = nn.SiLU()

    def forward(self, x):
        """
        前向传播（简化版）

        注意：这是Mamba块的简化实现，实际生产建议使用mamba_ssm官方库。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, sequence_length, dim)
        """
        b, l, d = x.shape  # 批量大小, 序列长度, 维度

        # 输入投影
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # 分割为两个部分

        # 1D卷积处理
        x = x.transpose(1, 2)  # (b, d_inner, l)
        x = self.conv1d(x)[:, :, :l].transpose(1, 2)  # 卷积并截断到原始长度

        # 激活函数
        x = self.act(x)

        # 简单门控机制模拟SSM输出（简化版）
        y = x * self.act(z)

        # 输出投影
        return self.out_proj(y)


class MambaBlock2D(nn.Module):
    """
    2D Mamba 块：将 Mamba 应用于 2D 特征图

    通过展平空间维度，将2D特征图转换为序列，应用Mamba块，
    然后恢复为2D格式。用于图像分割任务。

    参数:
        dim (int): 输入/输出通道数
        d_state (int): Mamba块的状态维度，默认16
    """

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.mamba = MambaBlock(dim, d_state=d_state)  # 1D Mamba块
        self.norm = nn.LayerNorm(dim)  # 层归一化

    def forward(self, x):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入2D特征图，形状为 (batch_size, channels, height, width)

        返回:
            torch.Tensor: 输出2D特征图，形状与输入相同
        """
        b, c, h, w = x.shape  # 批量大小, 通道数, 高度, 宽度

        # 展平空间维度：(B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).view(b, h * w, c)

        # 层归一化
        x_flat = self.norm(x_flat)

        # 应用Mamba块
        x_mamba = self.mamba(x_flat)

        # 还原回2D格式：(B, H*W, C) -> (B, C, H, W)
        return x_mamba.view(b, h, w, c).permute(0, 3, 1, 2)


class MambaDown(nn.Module):
    """下采样：MaxPool -> DoubleConv -> Mamba"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            MambaBlock2D(out_channels),
        )

    def forward(self, x):
        return self.main(x)


class MambaUp(nn.Module):
    """上采样：Upsample -> Concat -> DoubleConv -> Mamba"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
        self.mamba = MambaBlock2D(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 补齐尺寸差异（参考 model.py）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.mamba(x)


class MambaUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=64, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = MambaDown(base_channels, base_channels * 2)
        self.down2 = MambaDown(base_channels * 2, base_channels * 4)
        self.down3 = MambaDown(base_channels * 4, base_channels * 8)

        factor = 2 if bilinear else 1
        self.down4 = MambaDown(base_channels * 8, base_channels * 16 // factor)

        self.up1 = MambaUp(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = MambaUp(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = MambaUp(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = MambaUp(base_channels * 2, base_channels, bilinear)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_mamba_unet(in_channels=3, num_classes=2, **kwargs):
    return MambaUNet(in_channels=in_channels, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    model = build_mamba_unet(3, 2)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")  # 预期应为 (1, 2, 256, 256)
