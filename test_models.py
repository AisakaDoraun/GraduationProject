#!/usr/bin/env python3
"""
测试两个模型加载功能
"""

import torch
from src.model import build_unet
from src.model_mamba import build_mamba_unet


def test_model_loading():
    """测试两个模型是否能正确加载和推理"""

    print("🧪 测试模型加载...")

    # 测试U-Net模型
    print("\n1. 测试U-Net模型:")
    unet_model = build_unet(in_channels=3, num_classes=2, base_channels=64)
    print(f"   ✅ U-Net模型创建成功")
    print(f"   📊 参数数量: {sum(p.numel() for p in unet_model.parameters()):,}")

    # 测试Mamba U-Net模型
    print("\n2. 测试Mamba U-Net模型:")
    mamba_model = build_mamba_unet(in_channels=3, num_classes=2, base_channels=64)
    print(f"   ✅ Mamba U-Net模型创建成功")
    print(f"   📊 参数数量: {sum(p.numel() for p in mamba_model.parameters()):,}")

    # 测试推理
    print("\n3. 测试推理功能:")
    test_input = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        # U-Net推理
        unet_output = unet_model(test_input)
        print(f"   ✅ U-Net推理成功，输出形状: {unet_output.shape}")

        # Mamba U-Net推理
        mamba_output = mamba_model(test_input)
        print(f"   ✅ Mamba U-Net推理成功，输出形状: {mamba_output.shape}")

    print("\n🎉 所有测试通过！")
    print("📋 使用说明:")
    print(
        "   - 使用U-Net模型: python inference.py --model-type unet --checkpoint <path> --image <path>"
    )
    print(
        "   - 使用Mamba U-Net模型: python inference.py --model-type mamba --checkpoint <path> --image <path>"
    )


if __name__ == "__main__":
    test_model_loading()
