import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Callable
import random
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    """
    图像分割数据集类

    用于加载图像和对应的分割掩码，支持多种图像格式。
    自动匹配图像和掩码文件，支持数据增强和预处理。

    参数:
        image_dir (str): 图像文件目录路径
        mask_dir (str): 掩码文件目录路径
        transform (callable, optional): 图像和掩码的联合转换函数（同时应用于两者）
        image_transform (callable, optional): 仅应用于图像的转换函数
        mask_transform (callable, optional): 仅应用于掩码的转换函数
        image_extensions (tuple): 支持的图像文件扩展名，默认为常见图像格式
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_extensions = image_extensions

        self.image_paths = []
        self.mask_paths = []

        self._load_data()

    def _load_data(self):
        """
        加载图像和对应的掩码路径

        扫描图像目录，查找所有支持的图像文件，然后尝试在掩码目录中
        找到对应的掩码文件（相同文件名，不同扩展名）。

        抛出:
            FileNotFoundError: 如果图像或掩码目录不存在
        """
        # 检查目录是否存在
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")

        # 获取所有支持的图像文件
        image_files = [
            f
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(self.image_extensions)
        ]
        image_files.sort()  # 按文件名排序确保可重复性

        # 为每个图像文件查找对应的掩码
        for img_file in image_files:
            img_path = os.path.join(self.image_dir, img_file)

            # 获取文件名（不含扩展名）
            base_name = os.path.splitext(img_file)[0]
            mask_path = None

            # 尝试所有支持的扩展名查找掩码文件
            for ext in self.image_extensions:
                candidate = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            # 如果未找到掩码，跳过该图像
            if mask_path is None:
                print(f"警告: 未找到 {img_file} 对应的掩码，跳过")
                continue

            # 保存匹配的图像-掩码对
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        """
        返回数据集大小

        返回:
            int: 数据集中图像-掩码对的数量
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的图像和掩码

        参数:
            idx (int): 数据索引

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 图像张量和掩码张量

        抛出:
            ValueError: 如果无法加载图像或掩码文件
        """
        # 获取文件路径
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 加载图像（转换为RGB格式）
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"无法加载图像 {img_path}: {e}")

        # 加载掩码（转换为灰度格式）
        try:
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            raise ValueError(f"无法加载掩码 {mask_path}: {e}")

        # 应用联合变换（如果提供）
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # 默认转换：图像转为张量，掩码转为长整型张量
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()
            # 如果掩码值大于1，进行二值化处理
            if mask.max() > 1:
                mask = (mask > 0).long()

        # 应用单独的图像变换（如果提供）
        if self.image_transform:
            image = self.image_transform(image)

        # 应用单独的掩码变换（如果提供）
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


class JointTransform:
    """
    联合变换类：对图像和掩码应用相同的几何变换

    确保图像和掩码在数据增强过程中保持空间对齐。
    支持随机裁剪、水平翻转、旋转和颜色抖动等增强操作。

    参数:
        image_size (Tuple[int, int]): 输出图像尺寸 (高度, 宽度)
        augment (bool): 是否启用数据增强
        num_classes (int): 分割类别数（用于掩码处理）
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        num_classes: int = 2,
    ):
        self.image_size = image_size  # 目标图像尺寸
        self.augment = augment  # 是否启用数据增强
        self.num_classes = num_classes  # 分割类别数

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.augment:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = transforms.functional.resized_crop(
                image, i, j, h, w, self.image_size
            )
            mask = transforms.functional.resized_crop(
                mask,
                i,
                j,
                h,
                w,
                self.image_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            )

            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle, fill=0)

            if random.random() > 0.5:
                cj = transforms.ColorJitter(
                    brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)
                )
                image = cj(image)
        else:
            image = transforms.functional.resize(image, self.image_size)
            mask = transforms.functional.resize(
                mask,
                self.image_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        mask = torch.from_numpy(np.array(mask)).long()
        if self.num_classes == 2:
            mask = (mask > 0).long()
        else:
            mask = torch.clamp(mask, 0, self.num_classes - 1)

        return image, mask


class SegmentationDataModule:
    """
    图像分割数据模块

    管理训练集、验证集、测试集的划分和数据加载器创建。
    提供完整的数据处理流水线，包括数据增强、批处理和并行加载。

    参数:
        image_dir (str): 图像文件目录路径
        mask_dir (str): 掩码文件目录路径
        image_size (Tuple[int, int]): 图像尺寸 (高度, 宽度)
        batch_size (int): 批量大小
        val_split (float): 验证集划分比例 (0.0-1.0)
        test_split (float): 测试集划分比例 (0.0-1.0)
        num_workers (int): 数据加载工作进程数（用于并行加载）
        seed (int): 随机种子（确保可重复性）
        augment (bool): 是否启用数据增强（仅对训练集）
        num_classes (int): 分割类别数（包括背景）
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 16,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 4,
        seed: int = 42,
        augment: bool = True,
        num_classes: int = 2,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.seed = seed
        self.augment = augment
        self.num_classes = num_classes

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare_data(self):
        """检查数据目录是否存在"""
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")

    def setup(self, stage: Optional[str] = None):
        """设置数据集划分"""
        full_dataset = SegmentationDataset(
            image_dir=self.image_dir, mask_dir=self.mask_dir
        )

        indices = list(range(len(full_dataset)))

        if self.test_split > 0:
            train_val_idx, test_idx = train_test_split(
                indices, test_size=self.test_split, random_state=self.seed
            )
        else:
            train_val_idx = indices
            test_idx = []

        if self.val_split > 0:
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=self.val_split, random_state=self.seed
            )
        else:
            train_idx = train_val_idx
            val_idx = []

        self.train_dataset = self._create_subset(full_dataset, train_idx, augment=True)
        self.val_dataset = self._create_subset(full_dataset, val_idx, augment=False)

        if test_idx:
            self.test_dataset = self._create_subset(
                full_dataset, test_idx, augment=False
            )

    def _create_subset(
        self, full_dataset: SegmentationDataset, indices: List[int], augment: bool
    ) -> Dataset:
        """创建带变换的子数据集"""
        transform = JointTransform(
            self.image_size,
            augment=augment and self.augment,
            num_classes=self.num_classes,
        )
        return SegmentationSubset(full_dataset, indices, transform)

    def train_dataloader(self) -> DataLoader:
        """获取训练数据加载器"""
        if self.train_dataset is None:
            self.setup()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """获取验证数据加载器"""
        if self.val_dataset is None:
            self.setup()

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """获取测试数据加载器"""
        if self.test_dataset is None:
            self.setup()

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return self.test_loader

    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            "image_dir": self.image_dir,
            "mask_dir": self.mask_dir,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "total_size": (len(self.train_dataset) if self.train_dataset else 0)
            + (len(self.val_dataset) if self.val_dataset else 0)
            + (len(self.test_dataset) if self.test_dataset else 0),
        }
        return info


class SegmentationSubset(Dataset):
    """分割数据集的子集"""

    def __init__(
        self, dataset: SegmentationDataset, indices: List[int], transform: Callable
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.dataset.image_paths[self.indices[idx]]
        mask_path = self.dataset.mask_paths[self.indices[idx]]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self.transform(image, mask)

        return image, mask


if __name__ == "__main__":
    data_module = SegmentationDataModule(
        image_dir="./data/images",
        mask_dir="./data/masks",
        image_size=(256, 256),
        batch_size=16,
        val_split=0.2,
        test_split=0.1,
        num_workers=4,
        augment=True,
        num_classes=2,
    )

    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    info = data_module.get_dataset_info()
    print("数据集信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if train_loader:
        images, masks = next(iter(train_loader))
        print(f"\n批次形状: images={images.shape}, masks={masks.shape}")
        print(f"图像范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"掩码唯一值: {torch.unique(masks).tolist()}")
