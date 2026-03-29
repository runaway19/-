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
import segmentation_models_pytorch as smp  # 【新增】用于加载你的分割模型
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
    output_dir = "C:/Users/Administrator/Desktop/PythonProject/cls_output"

    # 【新增】你的分割模型权重路径
    seg_model_path = "C:/Users/Administrator/Desktop/PythonProject/final_model_20260102_140556.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 384
    # 【显存抢救】因为加载了分割模型，显存占用大增。调小真实 Batch Size，增大累加步数
    batch_size = 4
    accumulate_steps = 8  # 4 * 8 = 等效 Batch Size 32

    epochs = 100
    lr_backbone = 1e-5
    lr_head = 1e-4
    num_classes = 5
    patience = 30


# ================= 1. 分类模型架构 =================
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
            new_conv.weight[:, :3] = old_conv.weight.clone() / 2.0
            new_conv.weight[:, 3:] = old_conv.weight.clone() / 2.0
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
            nn.Dropout(p=0.5),
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


# ================= 2. 端到端融合管线 (核心修改区) =================
class EndToEndPipeline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device

        # 1. 加载双分割模型
        logger.info("[*] 正在向管道注入分割模型...")
        self.model_ma = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1)
        self.model_rest = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=3)

        checkpoint = torch.load(cfg.seg_model_path, map_location=self.device)
        self.model_ma.load_state_dict(checkpoint['model_ma_state'])
        self.model_rest.load_state_dict(checkpoint['model_rest_state'])

        # 【关键】彻底冻结分割模型权重，节省显存，防止被分类任务带偏
        for param in self.model_ma.parameters(): param.requires_grad = False
        for param in self.model_rest.parameters(): param.requires_grad = False
        self.model_ma.eval()
        self.model_rest.eval()

        # 2. 加载分类模型
        self.cls_model = DeepDRTransformer(cfg.num_classes)

    def forward(self, img3):
        # 步骤 1：让输入原图穿过被冻结的分割模型，生成 Mask
        with torch.no_grad():
            # 推理时不用管模型处于什么模式，我们只要它的特征图
            pred_ma = torch.sigmoid(self.model_ma(img3))
            pred_rest = torch.sigmoid(self.model_rest(img3))
            mask4 = torch.cat([pred_ma, pred_rest], dim=1)  # (B, 4, H, W)

            # 【可选阈值二值化】如果想用纯黑白掩码，可以解开下面两行注释
            # threshold = torch.tensor([0.4, 0.5, 0.5, 0.4], device=img3.device).view(1, 4, 1, 1)
            # mask4 = (mask4 > threshold).float()

        # 步骤 2：自动拼接 6 通道
        img_extra = mask4[:, :3, :, :]
        img6 = torch.cat([img3, img_extra], dim=1)

        # 步骤 3：喂给分类模型，只训练这里
        return self.cls_model(img6, mask4)


# ================= 3. 数据集类 (已简化，无需读Mask) =================
class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['Image name']
        label = row['Retinopathy grade']

        img_path = os.path.join(self.img_dir, fname + ".jpg")
        image = cv2.imread(img_path)

        # 【极其重要】加入分割代码里的 CLAHE 预处理，否则你的分割模型认不出来原图！
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        image_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        if self.transform:
            augmented = self.transform(image=image_rgb)
            img_tensor = augmented['image']
        else:
            img_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label


# ================= 4. 训练与验证 =================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, accumulate_steps):
    model.train()
    # 强制分割模型保持 eval 状态
    model.model_ma.eval()
    model.model_rest.eval()

    train_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training", leave=False)

    for i, (img3, labels) in enumerate(pbar):
        img3, labels = img3.to(device), labels.to(device)

        with autocast(device_type='cuda'):
            outputs = model(img3)
            loss = criterion(outputs, labels)
            loss = loss / accumulate_steps

        scaler.scale(loss).backward()

        if ((i + 1) % accumulate_steps == 0) or (i + 1 == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulate_steps
        pbar.set_postfix({"loss": f"{loss.item() * accumulate_steps:.4f}"})

    return train_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    for img3, labels in loader:
        img3, labels = img3.to(device), labels.to(device)

        with autocast(device_type='cuda'):
            outputs = model(img3)
            loss = criterion(outputs, labels)

        val_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return val_loss / len(loader), acc, qwk


# ================= 5. 绘图与评估 =================
def plot_metrics(cfg, train_losses, val_losses, val_accs, val_qwks):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

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

    model = EndToEndPipeline(cfg).to(cfg.device)

    if os.path.exists(model_path):
        # 我们保存的时候只存了 cls_model 的权重，所以要对应加载
        model.cls_model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    else:
        logger.error("未找到最佳模型文件！")
        return

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img3, labels in tqdm(test_loader, desc="Testing"):
            img3 = img3.to(cfg.device)
            with autocast(device_type='cuda'):
                outputs = model(img3)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    logger.info(f"评估完成 -> Accuracy: {acc:.4f} | QWK: {qwk:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, zero_division=0))
    logger.info(f"\n混淆矩阵:\n{confusion_matrix(all_labels, all_preds)}")
    logger.info("=" * 50)


# ================= 6. 主程序 =================
def main():
    seed_everything(42)
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_model_path = os.path.join(cfg.output_dir, "best_model.pth")

    df = pd.read_csv(cfg.train_label_py)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(df))
    train_df, val_df = df[:split], df[split:]

    # 注意这里的 Normalize 必须使用预训练的 ImageNet 标准，才能让分割和分类模型都舒服
    train_tsfm = A.Compose([
        A.Resize(height=cfg.img_size, width=cfg.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_tsfm = A.Compose([
        A.Resize(height=cfg.img_size, width=cfg.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 不再需要传 cfg.train_mask_dir
    train_ds = DRDataset(train_df, cfg.train_img_dir, train_tsfm)
    val_ds = DRDataset(val_df, cfg.train_img_dir, val_tsfm)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 实例化整个管线
    pipeline_model = EndToEndPipeline(cfg).to(cfg.device)

    criterion = nn.CrossEntropyLoss()

    # 【极其重要】优化器里只能放入分类模型（cls_model）的参数，分割模型参数已被冻结，不参与更新
    optimizer = optim.AdamW([
        {'params': pipeline_model.cls_model.img_backbone.parameters(), 'lr': cfg.lr_backbone},
        {'params': pipeline_model.cls_model.area_embed.parameters(), 'lr': cfg.lr_head},
        {'params': pipeline_model.cls_model.classifier.parameters(), 'lr': cfg.lr_head},
    ], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    scaler = GradScaler()

    best_qwk = -1
    epochs_no_improve = 0

    history_train_loss, history_val_loss = [], []
    history_val_acc, history_val_qwk = [], []

    logger.info("[*] 端到端 (End-to-End) 训练管线启动！")
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(pipeline_model, train_loader, optimizer, criterion, scaler, cfg.device,
                                     cfg.accumulate_steps)
        val_loss, val_acc, val_qwk = validate_one_epoch(pipeline_model, val_loader, criterion, cfg.device)

        scheduler.step(val_qwk)

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)
        history_val_qwk.append(val_qwk)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch + 1}/{cfg.epochs}] | LR: {current_lr:.2e} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            epochs_no_improve = 0
            # 只保存分类模型的权重
            torch.save(pipeline_model.cls_model.state_dict(), best_model_path)
            logger.info(f"  [+] 更好模型已保存! 当前最佳 QWK: {best_qwk:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                logger.info(f"[!] 连续 {cfg.patience} 个 Epoch 未提升，触发早停。")
                break

        torch.cuda.empty_cache()

    plot_metrics(cfg, history_train_loss, history_val_loss, history_val_acc, history_val_qwk)
    evaluate_model(cfg, val_loader, best_model_path)


if __name__ == "__main__":
    main()