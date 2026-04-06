import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path

from src.model import build_unet
from src.model_mamba import build_mamba_unet
from src.database import JointTransform


def load_model(checkpoint_path, device="cpu", model_type="unet"):
    """
    加载训练好的模型

    从检查点文件加载模型架构和权重，并将模型设置为评估模式。

    参数:
        checkpoint_path (str): 检查点文件路径
        device (str): 加载设备（"cpu"或"cuda"）
        model_type (str): 模型类型，"unet"或"mamba"

    返回:
        nn.Module: 加载好的模型
    """
    # 加载检查点文件
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从检查点获取模型配置参数
    num_classes = checkpoint.get("num_classes", 2)  # 分割类别数，默认为2
    base_channels = checkpoint.get("base_channels", 64)  # 基础通道数，默认为64

    # 根据模型类型创建模型
    if model_type == "mamba":
        model = build_mamba_unet(
            in_channels=3,  # 输入通道数（RGB图像）
            num_classes=num_classes,  # 分割类别数
            base_channels=base_channels,  # 基础通道数
        )
        print(
            f"📦 加载 Mamba U-Net 模型 (通道数={base_channels}, 类别数={num_classes})"
        )
    else:
        model = build_unet(
            in_channels=3,  # 输入通道数（RGB图像）
            num_classes=num_classes,  # 分割类别数
            bilinear=True,  # 使用双线性插值上采样
            base_channels=base_channels,  # 基础通道数
        )
        print(f"📦 加载 U-Net 模型 (通道数={base_channels}, 类别数={num_classes})")

    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 将模型移动到指定设备
    model.to(device)

    # 设置为评估模式（禁用Dropout和BatchNorm的统计更新）
    model.eval()

    return model


def preprocess_image(image_path, image_size=256):
    """
    预处理图像

    加载图像并应用与训练时相同的预处理流程，包括尺寸调整和归一化。

    参数:
        image_path (str): 输入图像文件路径
        image_size (int): 目标图像尺寸（正方形）

    返回:
        torch.Tensor: 预处理后的图像张量，形状为 (1, 3, image_size, image_size)
    """
    # 创建与训练时相同的变换（禁用数据增强）
    transform = JointTransform(
        image_size=(image_size, image_size),  # 目标尺寸
        augment=False,  # 推理时不使用数据增强
        num_classes=2,  # 类别数（用于掩码处理）
    )

    # 加载图像并转换为RGB格式
    image = Image.open(image_path).convert("RGB")

    # 创建一个空的掩码（推理时不需要真实掩码）
    dummy_mask = Image.new("L", image.size, 0)

    # 应用变换（只使用图像部分）
    image_tensor, _ = transform(image, dummy_mask)

    # 添加批次维度并返回
    return image_tensor.unsqueeze(0)


def predict(model, image_tensor, device="cpu"):
    """
    进行预测

    使用训练好的模型对输入图像进行分割预测。

    参数:
        model: 训练好的分割模型
        image_tensor (torch.Tensor): 预处理后的图像张量
        device (str): 推理设备

    返回:
        tuple: (预测掩码numpy数组, 类别概率numpy数组)
    """
    with torch.no_grad():  # 禁用梯度计算，节省内存
        # 将图像移动到指定设备
        image_tensor = image_tensor.to(device)

        # 前向传播获取logits
        logits = model(image_tensor)

        # 将logits转换为概率（softmax）
        probs = F.softmax(logits, dim=1)

        # 获取预测类别（取最大概率的类别）
        preds = torch.argmax(probs, dim=1)

    # 移除批次维度并转换为numpy数组
    return preds.squeeze(0).cpu().numpy(), probs.squeeze(0).cpu().numpy()


def save_results(original_image, prediction, output_path):
    """保存结果"""
    # 将预测掩码调整到原始图像尺寸
    original_size = original_image.size  # (width, height)
    prediction_resized = Image.fromarray(prediction.astype(np.uint8))
    prediction_resized = prediction_resized.resize(
        original_size, Image.Resampling.NEAREST
    )
    prediction_array = np.array(prediction_resized)

    # 保存预测掩码
    mask_path = Path(output_path).with_suffix(".mask.png")
    mask_image = Image.fromarray((prediction_array * 255).astype(np.uint8))
    mask_image.save(mask_path)
    print(f"掩码已保存至: {mask_path}")

    # 保存原始图像和掩码的叠加
    overlay_path = Path(output_path).with_suffix(".overlay.png")
    original_array = np.array(original_image)

    # 创建叠加图像（红色表示预测区域）
    overlay = original_array.copy()
    if prediction_array.max() > 0:
        mask_indices = prediction_array > 0
        overlay[mask_indices] = [255, 0, 0]  # 红色

    overlay_image = Image.fromarray(overlay)
    overlay_image.save(overlay_path)
    print(f"叠加图像已保存至: {overlay_path}")

    # 保存原始图像
    original_path = Path(output_path).with_suffix(".original.png")
    original_image.save(original_path)
    print(f"原始图像已保存至: {original_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用训练好的U-Net模型进行图像分割推理"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点文件路径")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--output", type=str, default="prediction_result.png", help="输出图像路径"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cuda", "cpu"], help="推理设备"
    )
    parser.add_argument("--image-size", type=int, default=256, help="图像大小")
    parser.add_argument(
        "--model-type",
        type=str,
        default="unet",
        choices=["unet", "mamba"],
        help="模型类型：unet 或 mamba",
    )

    args = parser.parse_args()

    # 设置设备
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        device_str = "cuda"
        print(f"使用 GPU 进行推理")
    else:
        device = torch.device("cpu")
        device_str = "cpu"
        print(f"使用 CPU 进行推理")

    # 加载模型
    print(f"📂 加载模型: {args.checkpoint}")
    print(f"🔧 模型类型: {args.model_type}")
    model = load_model(args.checkpoint, device_str, args.model_type)
    print(f"✅ 模型加载完成")

    # 预处理图像
    print(f"处理图像: {args.image}")
    image_tensor = preprocess_image(args.image, args.image_size)

    # 进行预测
    print("进行预测...")
    prediction, probabilities = predict(model, image_tensor, device_str)

    # 统计预测结果
    unique_classes, counts = np.unique(prediction, return_counts=True)
    print(f"\n预测结果统计:")
    print(f"图像尺寸: {prediction.shape}")
    print(f"类别分布:")
    for cls, count in zip(unique_classes, counts):
        percentage = count / prediction.size * 100
        print(f"  类别 {cls}: {count} 像素 ({percentage:.1f}%)")

    # 加载原始图像
    original_image = Image.open(args.image).convert("RGB")

    # 保存结果
    save_results(original_image, prediction, args.output)

    print(f"\n推理完成！")


if __name__ == "__main__":
    main()
