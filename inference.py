import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path

from src.model import build_unet
from src.database import JointTransform


def load_model(checkpoint_path, device="cpu"):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从检查点获取模型配置
    num_classes = checkpoint.get("num_classes", 2)
    base_channels = checkpoint.get("base_channels", 64)

    # 创建模型
    model = build_unet(
        in_channels=3,
        num_classes=num_classes,
        bilinear=True,
        base_channels=base_channels,
    )

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def preprocess_image(image_path, image_size=256):
    """预处理图像"""
    transform = JointTransform(
        image_size=(image_size, image_size), augment=False, num_classes=2
    )

    image = Image.open(image_path).convert("RGB")
    # 创建一个空的掩码（推理时不需要）
    dummy_mask = Image.new("L", image.size, 0)

    image_tensor, _ = transform(image, dummy_mask)
    return image_tensor.unsqueeze(0)  # 添加批次维度


def predict(model, image_tensor, device="cpu"):
    """进行预测"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

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
    print(f"加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, device_str)
    print(f"模型加载完成，类别数: {model.num_classes}")

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
