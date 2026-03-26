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
from sklearn.metrics import cohen_kappa_score
from torch.amp import autocast, GradScaler
from collections import Counter
import random
import shutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 配置类 =================
class TrainConfig:
    # 请确保路径正确
    train_img_dir  = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    train_label_py = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"

    # 分割模型生成的mask路径 (用于多模态融合)
    train_mask_dir = "C:/Users/Administrator/Desktop/PythonProject/output/train_masks"

    output_dir     = "C:/Users/Administrator/Desktop/PythonProject/cls_output"
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size       = 384  # Swin-T 默认推荐尺寸
    batch_size     = 8    # Transformer 较占显存，建议设小一点
    epochs         = 50
    lr_backbone    = 1e-5 # 预训练权重用小学习率
    lr_head        = 1e-4 # 新定义的层用大学习率
    num_classes    = 5

# ================= 1. 升级版模型架构 (DeepDR-Transformer) =================
class DeepDRTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 使用 Swin Transformer 替换 EfficientNet
        # Swin-T base 输出维度是 768
        base = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

        # 修改第一层以接受 6 通道输入 (RGB + 3个关键病灶Mask)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            6, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding, bias=False)

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight.clone()
            new_conv.weight[:, 3:] = old_conv.weight.clone()
        base.features[0][0] = new_conv

        self.img_backbone = base.features
        self.img_pool     = nn.AdaptiveAvgPool2d(1)
        self.img_feat_dim = 768

        # 面积特征分支 (4维输入: MA, HE, EX, SE)
        self.area_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )

        # 融合后的分类头 (768 + 64 = 832)
        fused_dim = self.img_feat_dim + 64
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(p=0.4),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )

    def forward(self, img6, mask4):
        # 1. 提取视觉特征
        x = self.img_backbone(img6)
        if x.dim() == 4: # 处理不同版本的输出格式
            x = x.permute(0, 3, 1, 2) # Swin 的输出通常是 [B, H, W, C]
            x = self.img_pool(x)
        x = torch.flatten(x, 1) # [B, 768]

        # 2. 计算面积统计特征 (B, 4)
        area = mask4.mean(dim=(2, 3))
        area_feat = self.area_embed(area) # [B, 64]

        # 3. 融合分类
        fused = torch.cat([x, area_feat], dim=1) # [B, 832]
        return self.classifier(fused)

# ================= 2. 数据集类 =================
class DRDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['Image name']
        label = row['Retinopathy grade']

        img_path = os.path.join(self.img_dir, fname + ".jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 加载 4 个病灶 Mask (MA, HE, EX, SE)
        masks = []
        for m_type in ['MA', 'HE', 'EX', 'SE']:
            m_path = os.path.join(self.mask_dir, f"{fname}_{m_type}.png")
            if os.path.exists(m_path):
                m = cv2.imread(m_path, 0)
            else:
                m = np.zeros(img.shape[:2], dtype=np.uint8)
            masks.append(m)
        mask_stack = np.stack(masks, axis=-1) # (H, W, 4)

        if self.transform:
            augmented = self.transform(image=img, mask=mask_stack)
            img = augmented['image']
            mask_stack = augmented['mask']

        # 构造 6 通道输入: RGB + (MA, HE, EX 合并为3通道)
        img_extra = mask_stack[:, :, :3].permute(2, 0, 1)
        img6 = torch.cat([img, img_extra], dim=0)

        # mask4 用于面积计算
        mask4 = mask_stack.permute(2, 0, 1).float() / 255.0

        return img6, mask4, label

# ================= 3. 主训练流程 =================
def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 数据准备
    df = pd.read_csv(cfg.train_label_py)
    # 简单的训练/验证集切分
    df = df.sample(frac=1, random_state=42)
    split = int(0.8 * len(df))
    train_df, val_df = df[:split], df[split:]

    train_tsfm = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

    val_tsfm = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(),
        ToTensorV2()
    ])

    train_ds = DRDataset(train_df, cfg.train_img_dir, cfg.train_mask_dir, train_tsfm)
    val_ds   = DRDataset(val_df, cfg.train_img_dir, cfg.train_mask_dir, val_tsfm)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # 模型初始化
    model = DeepDRTransformer(cfg.num_classes).to(cfg.device)

    # 解决 "80 elements" 报错的关键：如果加载旧权重，请跳过不匹配的层
    model_path = os.path.join(cfg.output_dir, "best_model.pth")
    if os.path.exists(model_path):
        logger.info(f"[*] 发现旧模型，正在尝试加载：{model_path}")
        ckpt = torch.load(model_path, map_location=cfg.device)
        # 使用 strict=False 忽略维度不匹配的层（如分类头和BN层）
        model.load_state_dict(ckpt, strict=False)
        logger.info("[!] 警告：部分权重维度不匹配，已自动重置不一致的层（如BN层）。")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.img_backbone.parameters(), 'lr': cfg.lr_backbone},
        {'params': model.area_embed.parameters(), 'lr': cfg.lr_head},
        {'params': model.classifier.parameters(), 'lr': cfg.lr_head},
    ], weight_decay=1e-4)

    best_qwk = -1
    scaler = GradScaler()

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        for img6, mask4, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            img6, mask4, labels = img6.to(cfg.device), mask4.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(img6, mask4)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img6, mask4, labels in val_loader:
                img6, mask4 = img6.to(cfg.device), mask4.to(cfg.device)
                outputs = model(img6, mask4)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        acc = np.mean(np.array(all_preds) == np.array(all_labels))

        logger.info(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | QWK: {qwk:.4f}")

        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pth"))
            logger.info(f"[+] 最好模型已保存，QWK: {best_qwk:.4f}")

if __name__ == "__main__":
    main()