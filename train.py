import os
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import sys

from src.database import SegmentationDataModule
from src.model import build_unet


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = (
            nn.functional.one_hot(targets, num_classes=logits.shape[1])
            .permute(0, 3, 1, 2)
            .float()
        )

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice loss"""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


def compute_iou(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute mean IoU"""
    preds = torch.argmax(logits, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(torch.tensor(1.0, device=logits.device))
    return torch.mean(torch.stack(ious))


def compute_dice_score(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute mean Dice score"""
    preds = torch.argmax(logits, dim=1)
    dices = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().float()
        total = pred_cls.sum().float() + target_cls.sum().float()
        if total > 0:
            dices.append(2.0 * intersection / total)
        else:
            dices.append(torch.tensor(1.0, device=logits.device))
    return torch.mean(torch.stack(dices))


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0

    epoch_str = f"[{epoch + 1:03d}/{total_epochs:03d}]"

    pbar = tqdm(
        loader,
        desc=f"🚀 训练中 {epoch_str}",
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        iou = compute_iou(logits, masks, model.num_classes)
        dice = compute_dice_score(logits, masks, model.num_classes)

        running_loss += loss.item()
        running_iou += iou.item()
        running_dice += dice.item()
        num_batches += 1

        if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "IoU": f"{iou.item():.4f}",
                    "Dice": f"{dice.item():.4f}",
                }
            )

    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    avg_dice = running_dice / num_batches

    return {
        "loss": avg_loss,
        "iou": avg_iou,
        "dice": avg_dice,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0

    pbar = tqdm(
        loader,
        desc="📊 验证中",
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        iou = compute_iou(logits, masks, model.num_classes)
        dice = compute_dice_score(logits, masks, model.num_classes)

        running_loss += loss.item()
        running_iou += iou.item()
        running_dice += dice.item()
        num_batches += 1

        if batch_idx % 5 == 0 or batch_idx == len(loader) - 1:
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "IoU": f"{iou.item():.4f}",
                    "Dice": f"{dice.item():.4f}",
                }
            )

    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    avg_dice = running_dice / num_batches

    return {
        "loss": avg_loss,
        "iou": avg_iou,
        "dice": avg_dice,
    }


def save_checkpoint(
    model, optimizer, scheduler, epoch, metrics, save_dir, is_best=False
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }

    torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}.pt")
    if is_best:
        torch.save(checkpoint, save_dir / "best_checkpoint.pt")


def main(args):
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    data_module = SegmentationDataModule(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=args.augment,
        num_classes=args.num_classes,
    )

    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = build_unet(
        in_channels=3,
        num_classes=args.num_classes,
        bilinear=args.bilinear,
        base_channels=args.base_channels,
        dropout=args.dropout,
    )
    model.num_classes = args.num_classes
    model = model.to(device)

    if args.loss == "combined":
        criterion = CombinedLoss(dice_weight=args.dice_weight, ce_weight=args.ce_weight)
    elif args.loss == "dice":
        criterion = DiceLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        scheduler = None

    start_epoch = 0
    best_val_iou = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_iou = checkpoint["metrics"].get("iou", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val IoU: {best_val_iou:.4f}")

    print("\n" + "=" * 70)
    print("🎯 训练配置")
    print("=" * 70)
    print(f"  📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"  ⚙️  设备: {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
    )
    print(f"  🔄 训练轮数: {args.epochs}")
    print(f"  📦 批次大小: {args.batch_size}")
    print(
        f"  📚 训练样本数: {len(data_module.train_dataset) if data_module.train_dataset else 0}"
    )
    print(
        f"  🧪 验证样本数: {len(data_module.val_dataset) if data_module.val_dataset else 0}"
    )
    print(f"  📈 学习率: {args.lr}")
    print(f"  📉 损失函数: {args.loss}")
    print(f"  ⚡ 优化器: {args.optimizer}")
    print(f"  🔧 学习率调度器: {args.scheduler}")
    print(f"  🏗️  模型: U-Net (通道数={args.base_channels}, 类别数={args.num_classes})")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        epoch_str = f"[{epoch + 1:03d}/{args.epochs:03d}]"

        print(f"\n{'━' * 70}")
        print(f"📈 第 {epoch_str} 轮训练 - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'━' * 70}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        val_metrics = validate(model, val_loader, criterion, device)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # 格式化指标
        train_loss_str = f"{train_metrics['loss']:.4f}"
        train_iou_str = f"{train_metrics['iou']:.4f}"
        train_dice_str = f"{train_metrics['dice']:.4f}"
        val_loss_str = f"{val_metrics['loss']:.4f}"
        val_iou_str = f"{val_metrics['iou']:.4f}"
        val_dice_str = f"{val_metrics['dice']:.4f}"

        print(f"\n📊 第 {epoch_str} 轮结果")
        print(f"{'─' * 40}")
        print(f"  🚀 训练集:")
        print(f"    • 损失值:    {train_loss_str}")
        print(f"    • IoU:      {train_iou_str}")
        print(f"    • Dice系数: {train_dice_str}")
        print(f"  🧪 验证集:")
        print(f"    • 损失值:    {val_loss_str}")
        print(f"    • IoU:      {val_iou_str}")
        print(f"    • Dice系数: {val_dice_str}")
        print(f"  ⚙️  设置:")
        print(f"    • 学习率:    {current_lr:.6f}")
        print(f"    • 耗时:      {elapsed:.1f}秒")

        is_best = val_metrics["iou"] > best_val_iou
        if is_best:
            best_val_iou = val_metrics["iou"]
            print(f"\n🎉 🎉 🎉 新的最佳验证集 IoU: {best_val_iou:.4f} 🎉 🎉 🎉")

        if (epoch + 1) % args.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, args.save_dir, is_best
            )

    print(f"\n{'=' * 70}")
    print(f"✅ 训练完成")
    print(f"{'=' * 70}")
    print(f"  🏆 最佳验证集 IoU: {best_val_iou:.4f}")
    print(f"  📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  💾 检查点保存至: {args.save_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for image segmentation")

    parser.add_argument(
        "--image-dir", type=str, default="./data/images", help="Path to image directory"
    )
    parser.add_argument(
        "--mask-dir", type=str, default="./data/masks", help="Path to mask directory"
    )
    parser.add_argument(
        "--num-classes", type=int, default=2, help="Number of segmentation classes"
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image size (square)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "none"],
        help="LR scheduler",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="combined",
        choices=["combined", "dice", "ce"],
        help="Loss function",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
        help="Weight for Dice loss in combined loss",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.5,
        help="Weight for CE loss in combined loss",
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=True, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--base-channels", type=int, default=64, help="Base number of channels"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--augment", action="store_true", default=True, help="Use data augmentation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint save directory",
    )
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )

    args = parser.parse_args()
    main(args)
