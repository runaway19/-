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
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from torch.amp import autocast, GradScaler
import random
import matplotlib.pyplot as plt

# ================= 基础设置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= 配置类 =================
class TrainConfig:
    train_img_dir = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    train_label_py = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    train_mask_dir = "C:/Users/Administrator/Desktop/PythonProject/output/train_masks"
    output_dir = "C:/Users/Administrator/Desktop/PythonProject/cls_output"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 384
    batch_size = 8
    epochs = 100
    lr_backbone = 1e-5
    lr_head = 1e-4
    num_classes = 5
    patience = 30  # 稍微增加一点耐心


# ================= 1. 模型架构 =================
class DeepDRTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

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
        self.img_pool = nn.AdaptiveAvgPool2d(1)
        self.img_feat_dim = 768

        self.area_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )

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
        x = self.img_backbone(img6)
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
            x = self.img_pool(x)
        x = torch.flatten(x, 1)

        area = mask4.mean(dim=(2, 3))
        area_feat = self.area_embed(area)

        fused = torch.cat([x, area_feat], dim=1)
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

        masks = []
        for m_type in ['MA', 'HE', 'EX', 'SE']:
            m_path = os.path.join(self.mask_dir, f"{fname}_{m_type}.png")
            if os.path.exists(m_path):
                m = cv2.imread(m_path, 0)
            else:
                m = np.zeros(img.shape[:2], dtype=np.uint8)
            masks.append(m)
        mask_stack = np.stack(masks, axis=-1)

        if self.transform:
            augmented = self.transform(image=img, mask=mask_stack)
            img = augmented['image']
            mask_aug = augmented['mask']

            if isinstance(mask_aug, torch.Tensor):
                if mask_aug.ndim == 3 and mask_aug.shape[-1] == 4:
                    mask_aug = mask_aug.permute(2, 0, 1)
            else:
                mask_aug = torch.from_numpy(mask_aug.transpose(2, 0, 1))
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_aug = torch.from_numpy(mask_stack.transpose(2, 0, 1))

        mask4 = mask_aug.float() / 255.0
        img_extra = mask4[:3, :, :]
        img6 = torch.cat([img, img_extra], dim=0)

        return img6, mask4, label


# ================= 3. 训练与验证核心 =================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    train_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for img6, mask4, labels in pbar:
        img6, mask4, labels = img6.to(device), mask4.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(img6, mask4)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return train_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    for img6, mask4, labels in loader:
        img6, mask4, labels = img6.to(device), mask4.to(device), labels.to(device)

        outputs = model(img6, mask4)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return val_loss / len(loader), acc, qwk


# ================= 4. 绘图与评估 =================
def plot_metrics(cfg, train_losses, val_losses, val_accs, val_qwks):
    """绘制训练曲线并保存"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 准确率和 QWK 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, 'g-', label='Val Accuracy')
    plt.plot(epochs, val_qwks, 'm-', label='Val QWK')
    plt.title('Validation Accuracy & QWK')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(cfg.output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"[*] 训练曲线已保存至: {save_path}")


def evaluate_model(cfg, test_loader, model_path):
    logger.info("=" * 50)
    logger.info("[*] 正在加载最佳权重进行最终评估...")

    model = DeepDRTransformer(cfg.num_classes).to(cfg.device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    else:
        logger.error("未找到最佳模型文件！")
        return

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img6, mask4, labels in tqdm(test_loader, desc="Testing"):
            img6, mask4 = img6.to(cfg.device), mask4.to(cfg.device)
            outputs = model(img6, mask4)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    logger.info(f"评估完成 -> Accuracy: {acc:.4f} | QWK: {qwk:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, zero_division=0))
    logger.info(f"\n混淆矩阵:\n{confusion_matrix(all_labels, all_preds)}")
    logger.info("=" * 50)


# ================= 5. 主程序 =================
def main():
    seed_everything(42)
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_model_path = os.path.join(cfg.output_dir, "best_model.pth")

    df = pd.read_csv(cfg.train_label_py)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(df))
    train_df, val_df = df[:split], df[split:]

    # 稍微调小缩放比例，防止微小病灶被裁掉
    train_tsfm = A.Compose([
        A.RandomResizedCrop(size=(cfg.img_size, cfg.img_size), scale=(0.9, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

    val_tsfm = A.Compose([
        A.Resize(height=cfg.img_size, width=cfg.img_size),
        A.Normalize(),
        ToTensorV2()
    ])

    train_ds = DRDataset(train_df, cfg.train_img_dir, cfg.train_mask_dir, train_tsfm)
    val_ds = DRDataset(val_df, cfg.train_img_dir, cfg.train_mask_dir, val_tsfm)

    # 恢复正常的 DataLoader（移除会引发崩溃的 Sampler，恢复 shuffle）
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DeepDRTransformer(cfg.num_classes).to(cfg.device)

    # --- 核心改动：温和的类别权重损失 ---
    train_labels = train_df['Retinopathy grade'].values
    class_counts = np.bincount(train_labels, minlength=cfg.num_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)

    # 使用 1 / sqrt(count) 是一种更平滑的加权方式，既能照顾小类，又不会矫枉过正
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / np.sum(weights) * cfg.num_classes  # 归一化
    class_weights_tensor = torch.FloatTensor(weights).to(cfg.device)

    logger.info(f"[*] 使用平滑类别权重: {class_weights_tensor.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # 移除 label_smoothing

    optimizer = optim.AdamW([
        {'params': model.img_backbone.parameters(), 'lr': cfg.lr_backbone},
        {'params': model.area_embed.parameters(), 'lr': cfg.lr_head},
        {'params': model.classifier.parameters(), 'lr': cfg.lr_head},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_qwk = -1
    epochs_no_improve = 0

    # 记录曲线用的列表
    history_train_loss, history_val_loss = [], []
    history_val_acc, history_val_qwk = [], []

    logger.info("[*] 开始训练...")
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg.device)
        val_loss, val_acc, val_qwk = validate_one_epoch(model, val_loader, criterion, cfg.device)

        scheduler.step()

        # 记录数据用于画图
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)
        history_val_qwk.append(val_qwk)

        logger.info(f"Epoch [{epoch + 1}/{cfg.epochs}] | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  [+] 更好模型已保存! 当前最佳 QWK: {best_qwk:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                logger.info(f"[!] 连续 {cfg.patience} 个 Epoch 未提升，触发早停。")
                break

        torch.cuda.empty_cache()

    # 绘制并保存曲线
    plot_metrics(cfg, history_train_loss, history_val_loss, history_val_acc, history_val_qwk)

    # 最终评估
    evaluate_model(cfg, val_loader, best_model_path)


if __name__ == "__main__":
    main()