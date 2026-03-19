import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= 1. 日志与配置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainConfig:
    # --- 1. 图片所在的文件夹路径 ---
    train_img_dir = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    val_img_dir = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"

    # --- 2. 标签 CSV 文件路径 (请仔细核对这两个文件的实际名字！) ---
    train_csv = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    val_csv = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"

    # --- 3. 结果输出路径 ---
    output_dir = "C:/Users/ASUS/Desktop/PythonProject/cls_output"

    # --- 4. 训练超参数 ---
    num_classes = 5  # IDRiD 疾病评级是 0-4 级，共 5 类
    batch_size = 8  # 批次大小 (医疗图像较大，建议先用 8，如果显存报错改小为 4)
    epochs = 30  # 训练轮数
    learning_rate = 1e-4  # 学习率
    image_size = 512  # 图像统一缩放尺寸

    # --- 5. 运行设备 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 2. 数据预处理与 Dataset =================
def apply_clahe(image_bgr):
    """保持与分割任务一致的 CLAHE 图像增强"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


class IDRiDDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # 读取 CSV 文件
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 CSV 文件，请检查路径: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # IDRiD 数据集的 CSV 通常第一列是图像名，第二列是疾病等级 (Retinopathy grade)
        self.img_names = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1].values

        logger.info(f"成功加载 CSV: {csv_path}，共 {len(self.img_names)} 条记录")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 拼接图片名，补充 .jpg 后缀 (如果 CSV 里没有的话)
        img_name = str(self.img_names[idx])
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.tif')):
            img_name += '.jpg'

        img_path = os.path.join(self.img_dir, img_name)

        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"无法读取图片 (将被跳过): {img_path}")
            # 如果图片损坏或丢失，随机取另一张代替，防止训练中断
            return self.__getitem__((idx + 1) % len(self))

        # CLAHE 处理转 RGB
        image_rgb = apply_clahe(image)

        # 获取标签
        label = int(self.labels[idx])

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image_rgb)
            image_tensor = augmented['image']

        return image_tensor, torch.tensor(label, dtype=torch.long)


# ================= 3. 数据增强流水线 =================
def get_transforms(image_size):
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, val_transform


# ================= 4. 模型定义 =================
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 加载预训练的 EfficientNet-B4
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # 替换全连接层以适应我们的类别数 (5类)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ================= 5. 主训练循环 =================
def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. 准备数据
    train_tf, val_tf = get_transforms(cfg.image_size)

    logger.info("正在初始化训练集...")
    train_dataset = IDRiDDataset(cfg.train_csv, cfg.train_img_dir, transform=train_tf)

    logger.info("正在初始化验证集...")
    val_dataset = IDRiDDataset(cfg.val_csv, cfg.val_img_dir, transform=val_tf)

    # 显存或内存不足时，可以把 num_workers 改为 0 或者 2
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 初始化模型、损失函数和优化器
    logger.info(f"正在加载模型至设备: {cfg.device}...")
    model = EfficientNetClassifier(num_classes=cfg.num_classes).to(cfg.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    best_val_acc = 0.0

    # 3. 开始训练
    logger.info("开始训练...")
    for epoch in range(1, cfg.epochs + 1):
        # ---------- 训练阶段 ----------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{train_correct / train_total:.4f}"})

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(cfg.device), labels.to(cfg.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        logger.info(
            f"Epoch {epoch} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # ---------- 保存最佳模型 ----------
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_path = os.path.join(cfg.output_dir, "best_cls_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"★ 发现更高验证集准确率，已保存最佳模型至: {save_path}")

    logger.info("训练全部完成！")


if __name__ == "__main__":
    main()