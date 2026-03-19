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
    train_img_dir = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    val_img_dir = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    train_csv = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    val_csv = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    output_dir = "C:/Users/ASUS/Desktop/PythonProject/cls_output"

    num_classes = 5
    batch_size = 8
    epochs = 50  # 增加上限，靠早停控制
    learning_rate = 1e-4
    image_size = 512
    patience = 8  # 早停耐心值：连续8轮验证集不提升则停止
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 2. 数据处理 =================
def apply_clahe(image_bgr):
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
        self.df = pd.read_csv(csv_path)
        self.img_names = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1].values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = str(self.img_names[idx])
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_name += '.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image_rgb = apply_clahe(image)
        label = int(self.labels[idx])

        if self.transform:
            augmented = self.transform(image=image_rgb)
            image_tensor = augmented['image']

        return image_tensor, torch.tensor(label, dtype=torch.long)


# ================= 3. 增强版数据增强 =================
def get_transforms(image_size):
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # 新增：更强的几何变换
        A.Affine(translate_percent=0.1, scale=0.9, rotate=30, p=0.5),
        # 新增：更丰富的光影波动
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
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
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        num_features = self.model.classifier[1].in_features
        # 在全连接层前增加 Dropout 层，进一步抑制过拟合
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ================= 5. 训练主程序 =================
def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_tf, val_tf = get_transforms(cfg.image_size)
    train_dataset = IDRiDDataset(cfg.train_csv, cfg.train_img_dir, transform=train_tf)
    val_dataset = IDRiDDataset(cfg.val_csv, cfg.val_img_dir, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = EfficientNetClassifier(num_classes=cfg.num_classes).to(cfg.device)

    # 优化1：引入标签平滑 (Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-3)

    # 优化2：引入学习率调度器 (当指标不再提升时，学习率减半)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    early_stop_counter = 0

    logger.info("开始优化后的训练循环...")
    for epoch in range(1, cfg.epochs + 1):
        # --- 训练 ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # --- 验证 ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_acc = val_correct / val_total

        # 调度学习率
        scheduler.step(epoch_val_acc)

        logger.info(f"Epoch {epoch} | Train Acc: {train_correct / train_total:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # --- 优化3：早停逻辑与最佳模型保存 ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_cls_model.pth"))
            logger.info(f"★ 验证集准确率提升至 {best_val_acc:.4f}，模型已保存")
        else:
            early_stop_counter += 1
            if early_stop_counter >= cfg.patience:
                logger.warning(f"触发早停！连续 {cfg.patience} 轮无提升，训练提前结束。")
                break

    logger.info(f"训练结束，最佳验证集准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()