import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from datetime import datetime


# ================= 1. 配置区域 =================
class Config:
    # 路径配置
    train_csv = os.path.join("E:/DR_Classification/IDRiD_dataset/B._Disease_Grading/2._Groundtruths/"
                             "a. IDRiD_Disease Grading_Training Labels.csv")
    train_img_dir = os.path.join("E:/DR_Classification/IDRiD_dataset/B._Disease_Grading/1._Original_Images/original_data")
    mask_dir = os.path.join("E:/DR_Classification/IDRiD_dataset/B._Disease_Grading/1._Original_Images/reslut")
    log_file = "E:/DR_Classification/IDRiD_dataset/B._Disease_Grading/1._Original_Images/training_summary.log"

    # 训练参数
    model_name = "EfficientNet-B3 (7-Channel)"
    num_classes = 5
    img_size = 512
    batch_size = 8
    epochs = 30
    lr = 1e-4
    val_size = 0.2  # 验证集比例
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 病灶后缀
    mask_suffixes = ["_MA", "_HE", "_EX", "_SE"]


# ================= 2. 7通道数据加载器 =================
class DR7ChannelDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, transform=None):
        # 只保留需要的两列
        self.df = df[['Image name', 'Retinopathy grade']].reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 使用指定的列名
        img_id = str(self.df.iloc[idx]['Image name'])
        label = int(self.df.iloc[idx]['Retinopathy grade'])

        # 加载原图 (RGB)
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            image = cv2.imread(img_path.replace(".jpg", ".tif"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 加载 4 个病灶 Mask
        masks = []
        for suffix in Config.mask_suffixes:
            m_path = os.path.join(self.mask_dir, f"{img_id}{suffix}.png")
            if os.path.exists(m_path):
                mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.zeros(image.shape[:2], dtype=np.float32)
            masks.append(mask)

        # 数据增强与拼接 (3+4=7 通道)
        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            image = augmented['image']
            mask_tensors = [torch.from_numpy(m).unsqueeze(0) for m in augmented['masks']]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensors = [torch.from_numpy(m).unsqueeze(0) for m in masks]

        mask_stack = torch.cat(mask_tensors, dim=0)
        input_7ch = torch.cat([image, mask_stack], dim=0)

        return input_7ch, label


# ================= 3. 7通道分类模型 =================
class EfficientNet7Ch(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNet7Ch, self).__init__()
        self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        orig_conv = self.base_model.features[0][0]
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels=7,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )

        with torch.no_grad():
            self.base_model.features[0][0].weight[:, :3, :, :] = orig_conv.weight.clone()
            nn.init.kaiming_normal_(self.base_model.features[0][0].weight[:, 3:, :, :])

        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# ================= 4. 训练与验证逻辑 =================
def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return val_loss / len(loader), acc, kappa


def run_training():
    cfg = Config()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 4.1 数据划分与加载
    full_df = pd.read_csv(cfg.train_csv)
    train_df, val_df = train_test_split(full_df, test_size=cfg.val_size, random_state=cfg.seed,
                                        stratify=full_df['Retinopathy grade'])

    train_transform = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_loader = DataLoader(DR7ChannelDataset(train_df, cfg.train_img_dir, cfg.mask_dir, train_transform),
                              batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(DR7ChannelDataset(val_df, cfg.train_img_dir, cfg.mask_dir, val_transform),
                            batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    # 4.2 初始化
    model = EfficientNet7Ch(num_classes=cfg.num_classes).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_kappa = -1

    # 4.3 训练循环
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for images, labels in pbar:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss / len(train_loader))

        # 评估过程
        val_loss, val_acc, val_kappa = evaluate(model, val_loader, criterion, cfg.device)
        print(f"Validation - Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}")

        scheduler.step()
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), "best_model.pth")

    # 4.4 严格限制格式的日志保存
    with open(cfg.log_file, "a", encoding="utf-8") as f:
        f.write("-" * 30 + "\n")
        f.write(f"开始训练时间: {start_time}\n")
        f.write(
            f"所用超参数: LR={cfg.lr}, BatchSize={cfg.batch_size}, Epochs={cfg.epochs}, ImgSize={cfg.img_size}, ValSize={cfg.val_size}\n")
        f.write(f"模型名称: {cfg.model_name}\n")
        f.write(f"评估训练效果的指标: Best_Val_Quadratic_Kappa={best_kappa:.4f}, Last_Val_Acc={val_acc:.4f}\n")
        f.write(f"使用的数据量: Total={len(full_df)}, Train={len(train_df)}, Val={len(val_df)}\n")
        f.write("-" * 30 + "\n\n")


if __name__ == "__main__":
    run_training()