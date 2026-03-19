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
import matplotlib.pyplot as plt  # 新增：用于绘制曲线图

# ================= 1. 日志与配置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainConfig:
    # --- 1. 图片与标签路径 ---
    train_img_dir = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    val_img_dir = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    train_csv = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    val_csv = "C:/Users/ASUS/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    output_dir = "C:/Users/ASUS/Desktop/PythonProject/cls_output"

    # --- 2. 训练超参数 ---
    num_classes = 5
    batch_size = 8
    epochs = 50
    learning_rate = 1e-4
    image_size = 512
    label_smoothing = 0.2

    # --- 3. 运行设备 ---
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

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 CSV 文件，请检查路径: {csv_path}")

        df = pd.read_csv(csv_path)

        valid_img_names = []
        valid_labels = []

        for idx in range(len(df)):
            img_name = str(df.iloc[idx, 0])
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.tif')):
                img_name += '.jpg'

            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                valid_img_names.append(img_name)
                valid_labels.append(df.iloc[idx, 1])
            else:
                logger.warning(f"文件丢失，已从列表中移除: {img_path}")

        self.img_names = valid_img_names
        self.labels = valid_labels

        logger.info(f"成功加载 CSV: {csv_path}，有效记录共 {len(self.img_names)} 条")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_path)

        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        image_rgb = apply_clahe(image)
        label = int(self.labels[idx])

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
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        num_features = self.model.classifier[1].in_features
#        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x):
        return self.model(x)


# ================= 5. 绘图辅助函数 =================
def save_plots(history, output_dir):
    """根据历史记录绘制并保存训练过程曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 10))

    # 1. 绘制 Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 绘制 QWK (Quadratic Weighted Kappa)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_qwk'], label='Train QWK', marker='o')
    plt.plot(epochs, history['val_qwk'], label='Val QWK', marker='o')
    plt.title('Quadratic Weighted Kappa (QWK)')
    plt.xlabel('Epochs')
    plt.ylabel('QWK Score')
    plt.legend()
    plt.grid(True)

    # 3. 绘制 验证集准确率
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_acc'], label='Val Accuracy', color='green', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 4. 绘制 学习率变化
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['lr'], label='Learning Rate', color='orange', marker='o')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


# ================= 6. 主训练循环 =================
def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_tf, val_tf = get_transforms(cfg.image_size)
    train_dataset = IDRiDDataset(cfg.train_csv, cfg.train_img_dir, transform=train_tf)
    val_dataset = IDRiDDataset(cfg.val_csv, cfg.val_img_dir, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = EfficientNetClassifier(num_classes=cfg.num_classes).to(cfg.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler('cuda')

    best_val_qwk = -1.0

    # 【新增】：用于保存训练历史记录的字典
    history = {
        'train_loss': [], 'val_loss': [],
        'train_qwk': [], 'val_qwk': [],
        'val_acc': [], 'lr': []
    }

    logger.info("开始训练...")
    for epoch in range(1, cfg.epochs + 1):
        # ---------- 训练阶段 ----------
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_qwk = cohen_kappa_score(train_targets, train_preds, weights='quadratic')

        scheduler.step()

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(cfg.device), labels.to(cfg.device)

                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)

                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')
        epoch_val_acc = np.mean(np.array(val_preds) == np.array(val_targets))

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch} | LR: {current_lr:.6f} | "
            f"Train Loss: {epoch_train_loss:.4f} QWK: {epoch_train_qwk:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} QWK: {epoch_val_qwk:.4f} (Acc: {epoch_val_acc:.4f})"
        )

        # 【新增】：将当前 Epoch 的指标加入历史记录
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_qwk'].append(epoch_train_qwk)
        history['val_qwk'].append(epoch_val_qwk)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        # 【新增】：每次 Epoch 结束后自动保存最新折线图
        save_plots(history, cfg.output_dir)

        # ---------- 保存最佳模型 ----------
        if epoch_val_qwk > best_val_qwk:
            best_val_qwk = epoch_val_qwk
            save_path = os.path.join(cfg.output_dir, "best_cls_model_qwk.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"★ 发现更高验证集 QWK: {epoch_val_qwk:.4f}，已保存最佳模型至: {save_path}")

    logger.info("训练全部完成！你可以前往输出目录查看 training_history.png")


if __name__ == "__main__":
    main()