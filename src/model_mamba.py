import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MambaBlock(nn.Module):
    """
    Mamba 块：选择性状态空间模型

    参考: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    https://arxiv.org/abs/2312.00752
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank is None else dt_rank
        self.use_fast_path = use_fast_path

        # 输入投影
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=bias)

        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # 选择性SSM参数
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 dt
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # dt 偏置
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # 逆 softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A 参数（状态转移矩阵）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)

        # 激活函数
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        batch, seq_len, dim = x.shape

        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)  # 各 (B, L, D_inner)

        # 1D 卷积
        x = x.transpose(1, 2)  # (B, D_inner, L)
        x = self.conv1d(x)[:, :, :seq_len]  # 因果卷积
        x = x.transpose(1, 2)  # (B, L, D_inner)
        x = self.act(x)

        # 选择性SSM
        y = self.ssm(x, z)

        # 输出投影
        output = self.out_proj(y)
        return output

    def ssm(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """选择性状态空间模型"""
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        # 计算选择性参数
        xz = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(xz, [self.dt_rank, d_state, d_state], dim=-1)

        # dt 投影
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # A 参数
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()

        # 离散化
        dtA = torch.einsum("bld,dn->bldn", dt, A)  # (B, L, d_inner, d_state)
        dtB = torch.einsum(
            "bld,bldn->bldn", dt, B.unsqueeze(2)
        )  # (B, L, d_inner, d_state)

        # 扫描操作
        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        ys = []

        for i in range(seq_len):
            h = h * torch.exp(dtA[:, i]) + dtB[:, i] * x[:, i].unsqueeze(-1)
            y = torch.einsum("bdn,bdn->bd", h, C[:, i].unsqueeze(1)) + D * x[:, i]
            ys.append(y.unsqueeze(1))

        y = torch.cat(ys, dim=1)  # (B, L, d_inner)

        # 门控
        y = y * self.act(z)
        return y


class MambaConvBlock(nn.Module):
    """Mamba + 卷积块，用于图像特征提取"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        mamba_dim: int = 256,
        mamba_d_state: int = 16,
    ):
        super().__init__()

        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Mamba 层（处理空间信息）
        self.mamba = nn.Sequential(
            nn.Conv2d(out_channels, mamba_dim, 1),  # 投影到 Mamba 维度
            nn.ReLU(inplace=True),
            MambaBlock2D(mamba_dim, d_state=mamba_d_state),
            nn.Conv2d(mamba_dim, out_channels, 1),  # 投影回
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv(x)
        x = self.mamba(x)
        return x + identity


class MambaBlock2D(nn.Module):
    """2D Mamba 块，处理图像特征图"""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        patch_size: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.d_state = d_state

        # 将2D特征图展平为序列
        self.mamba_h = MambaBlock(dim, d_state=d_state)  # 水平方向
        self.mamba_v = MambaBlock(dim, d_state=d_state)  # 垂直方向

        # 层归一化
        self.norm_h = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            output: (B, C, H, W)
        """
        batch, channels, height, width = x.shape

        # 水平方向 Mamba
        x_h = x.permute(0, 2, 3, 1).reshape(
            batch * height, width, channels
        )  # (B*H, W, C)
        x_h = self.norm_h(x_h)
        x_h = self.mamba_h(x_h)
        x_h = x_h.reshape(batch, height, width, channels).permute(
            0, 3, 1, 2
        )  # (B, C, H, W)

        # 垂直方向 Mamba
        x_v = x.permute(0, 3, 2, 1).reshape(
            batch * width, height, channels
        )  # (B*W, H, C)
        x_v = self.norm_v(x_v)
        x_v = self.mamba_v(x_v)
        x_v = x_v.reshape(batch, width, height, channels).permute(
            0, 3, 2, 1
        )  # (B, C, H, W)

        # 合并两个方向
        output = x_h + x_v + x
        return output


class MambaUNet(nn.Module):
    """
    Mamba-UNet: 结合 Mamba 和 U-Net 的图像分割模型

    Args:
        in_channels: 输入通道数
        num_classes: 分割类别数
        base_channels: 基础通道数
        mamba_dims: 各层 Mamba 的维度
        mamba_d_state: Mamba 状态维度
        bilinear: 是否使用双线性上采样
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 64,
        mamba_dims: Tuple[int, ...] = (256, 512, 1024, 2048),
        mamba_d_state: int = 16,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(
            base_channels, base_channels * 2, mamba_dims[0], mamba_d_state
        )
        self.down2 = Down(
            base_channels * 2, base_channels * 4, mamba_dims[1], mamba_d_state
        )
        self.down3 = Down(
            base_channels * 4, base_channels * 8, mamba_dims[2], mamba_d_state
        )
        self.down4 = Down(
            base_channels * 8, base_channels * 16, mamba_dims[3], mamba_d_state
        )

        # 解码器
        factor = 2 if bilinear else 1
        self.up1 = Up(
            base_channels * 16,
            base_channels * 8 // factor,
            bilinear,
            mamba_dims[2],
            mamba_d_state,
        )
        self.up2 = Up(
            base_channels * 8,
            base_channels * 4 // factor,
            bilinear,
            mamba_dims[1],
            mamba_d_state,
        )
        self.up3 = Up(
            base_channels * 4,
            base_channels * 2 // factor,
            bilinear,
            mamba_dims[0],
            mamba_d_state,
        )
        self.up4 = Up(
            base_channels * 2, base_channels, bilinear, base_channels, mamba_d_state
        )

        # 输出层
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """双卷积层"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """下采样层（带 Mamba）"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mamba_dim: int,
        mamba_d_state: int,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MambaConvBlock(
                in_channels,
                out_channels,
                mamba_dim=mamba_dim,
                mamba_d_state=mamba_d_state,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样层（带 Mamba）"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        mamba_dim: Optional[int] = None,
        mamba_d_state: int = 16,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = MambaConvBlock(
                in_channels,
                out_channels,
                mamba_dim=mamba_dim or out_channels,
                mamba_d_state=mamba_d_state,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = MambaConvBlock(
                in_channels,
                out_channels,
                mamba_dim=mamba_dim or out_channels,
                mamba_d_state=mamba_d_state,
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # 处理尺寸差异
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def build_mamba_unet(
    in_channels: int = 3,
    num_classes: int = 2,
    base_channels: int = 64,
    mamba_dims: Tuple[int, ...] = (256, 512, 1024, 2048),
    mamba_d_state: int = 16,
    bilinear: bool = True,
) -> MambaUNet:
    """
    构建 Mamba-UNet 模型

    Args:
        in_channels: 输入通道数
        num_classes: 分割类别数
        base_channels: 基础通道数
        mamba_dims: 各层 Mamba 维度
        mamba_d_state: Mamba 状态维度
        bilinear: 是否使用双线性上采样
    """
    model = MambaUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        mamba_dims=mamba_dims,
        mamba_d_state=mamba_d_state,
        bilinear=bilinear,
    )
    return model


if __name__ == "__main__":
    # 测试模型
    model = build_mamba_unet(
        in_channels=3,
        num_classes=2,
        base_channels=64,
        mamba_dims=(256, 512, 1024, 2048),
        mamba_d_state=16,
        bilinear=True,
    )

    print("Mamba-UNet 模型结构:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # 测试 MambaBlock
    print("\n测试 MambaBlock:")
    mamba = MambaBlock(dim=256, d_state=16)
    x_seq = torch.randn(2, 128, 256)  # (B, L, D)
    y_seq = mamba(x_seq)
    print(f"MambaBlock 输入: {x_seq.shape}")
    print(f"MambaBlock 输出: {y_seq.shape}")

    # 测试 MambaBlock2D
    print("\n测试 MambaBlock2D:")
    mamba2d = MambaBlock2D(dim=256, d_state=16)
    x_img = torch.randn(2, 256, 32, 32)  # (B, C, H, W)
    y_img = mamba2d(x_img)
    print(f"MambaBlock2D 输入: {x_img.shape}")
    print(f"MambaBlock2D 输出: {y_img.shape}")
