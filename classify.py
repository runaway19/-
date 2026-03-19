"""
彻底修复LR跳变：全程只用一个OneCycleLR，不重建。

冻结期：backbone参数的requires_grad=False，梯度不传播
       但optimizer里包含backbone参数组（lr极小），scheduler照常计数

解冻后：backbone的requires_grad=True，开始实际更新
       LR按OneCycleLR曲线平滑衰减，无任何跳变

这是之前验证过可以工作的策略（那版Val QWK达到0.72的版本）。
"""

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
import matplotlib.pyplot as plt
from collections import Counter
import random
import shutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainConfig:
    train_img_dir = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    val_img_dir   = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    train_csv     = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    val_csv       = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    output_dir    = "C:/Users/Administrator/Desktop/PythonProject/cls_output"
    cache_train   = "C:/Users/Administrator/Desktop/PythonProject/cache_train_v2"
    cache_val     = "C:/Users/Administrator/Desktop/PythonProject/cache_val_v2"
    aug_cache     = "C:/Users/Administrator/Desktop/PythonProject/cache_augmented"

    num_classes   = 5
    batch_size    = 16
    epochs        = 60

    # 全程只有一组LR设置，不分阶段
    max_lr_head   = 5e-4      # 分类头峰值lr
    max_lr_bb     = 5e-5      # backbone峰值lr（冻结期不参与更新，解冻后自然生效）

    weight_decay  = 1e-2

    freeze_epochs = 10        # 前10轮backbone不传梯度

    early_stop_patience = 20
    min_delta           = 1e-4

    minority_target = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 预处理 =================
def ben_graham(image_bgr, size):
    img  = cv2.resize(image_bgr, (size, size))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=size/30)
    enh  = cv2.addWeighted(img, 4, blur, -4, 128)
    mask = np.zeros_like(enh)
    cv2.circle(mask, (size//2, size//2), int(size*0.45), (1,1,1), -1)
    return (enh * mask + 128*(1-mask)).astype(np.uint8)

def apply_clahe(img_rgb):
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(3.0, (8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

def make_6ch(image_bgr, size):
    orig = cv2.cvtColor(cv2.resize(image_bgr,(size,size)), cv2.COLOR_BGR2RGB)
    enh  = apply_clahe(ben_graham(image_bgr, size))
    return np.concatenate([orig, enh], axis=2).astype(np.uint8)

def build_cache(csv_path, img_dir, cache_dir, size):
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    n  = 0
    for i in tqdm(range(len(df)), desc=f"缓存→{os.path.basename(cache_dir)}"):
        name = str(df.iloc[i, 0])
        if not name.lower().endswith(('.jpg','.png','.jpeg','.tif')):
            name += '.jpg'
        dst = os.path.join(cache_dir, os.path.splitext(name)[0] + '.npy')
        if os.path.exists(dst): n+=1; continue
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None: continue
        np.save(dst, make_6ch(img, size))
        n += 1
    logger.info(f"缓存: {n}张 → {cache_dir}")


# ================= 少数类增强 =================
def build_minority_augmentation(csv_path, cache_dir, aug_dir, target, size):
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
    os.makedirs(aug_dir, exist_ok=True)

    df        = pd.read_csv(csv_path)
    cls_files = {i: [] for i in range(5)}
    for i in range(len(df)):
        name  = os.path.splitext(str(df.iloc[i,0]))[0] + '.npy'
        label = int(df.iloc[i,1])
        path  = os.path.join(cache_dir, name)
        if os.path.exists(path):
            cls_files[label].append(path)

    aug_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.9),
        A.Affine(translate_percent=0.05, scale=(0.85,1.15),
                 rotate=(-30,30), p=0.8),
        A.ColorJitter(brightness=0.35, contrast=0.35,
                      saturation=0.25, hue=0.06, p=0.8),
        A.RandomGamma(gamma_limit=(70,130), p=0.4),
        A.GaussianBlur(blur_limit=(3,7), p=0.3),
    ], additional_targets={'image2':'image'})

    records = []
    for cls_id, paths in cls_files.items():
        if len(paths) >= target: continue
        needed = target - len(paths)
        logger.info(f"类别{cls_id}: {len(paths)}张 → 增强{needed}张")
        for idx in range(needed):
            arr      = np.load(random.choice(paths))
            ch1, ch2 = arr[:,:,:3], arr[:,:,3:]
            aug      = aug_tf(image=ch1, image2=ch2)
            new_arr  = np.concatenate(
                [aug['image'], aug['image2']], axis=2).astype(np.uint8)
            fname    = f"aug_cls{cls_id}_{idx:04d}.npy"
            np.save(os.path.join(aug_dir, fname), new_arr)
            records.append((fname, cls_id))

    logger.info(f"增强完成: {len(records)}张")
    return records


# ================= Focal Loss =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce  = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ================= 模型：EfficientNet-B0 =================
class FundusClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base     = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
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

        num_features  = base.classifier[1].in_features  # 1280
        self.features = base.features
        self.avgpool  = base.avgpool

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        return self.classifier(torch.flatten(feat, 1))

    def set_backbone_grad(self, requires_grad: bool):
        for p in self.features.parameters():
            p.requires_grad = requires_grad


# ================= Dataset =================
class IDRiDDataset(Dataset):
    def __init__(self, csv_path, cache_dir, transform=None,
                 aug_dir=None, aug_records=None):
        self.transform = transform
        df     = pd.read_csv(csv_path)
        items, labels = [], []
        for i in range(len(df)):
            name = os.path.splitext(str(df.iloc[i,0]))[0] + '.npy'
            path = os.path.join(cache_dir, name)
            if os.path.exists(path):
                items.append(path)
                labels.append(int(df.iloc[i,1]))
        if aug_dir and aug_records:
            for (fname, lbl) in aug_records:
                items.append(os.path.join(aug_dir, fname))
                labels.append(lbl)
        self.items  = items
        self.labels = labels
        c = Counter(labels)
        logger.info("训练集分布:")
        for k in sorted(c):
            logger.info(f"  类别{k}: {c[k]}张")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        arr      = np.load(self.items[idx])
        ch1, ch2 = arr[:,:,:3], arr[:,:,3:]
        if self.transform:
            aug = self.transform(image=ch1, image2=ch2)
            t   = torch.cat([aug['image'], aug['image2']], dim=0)
        else:
            t = torch.zeros(6, arr.shape[0], arr.shape[1])
        return t, torch.tensor(self.labels[idx], dtype=torch.long)


# ================= 数据增强 =================
def get_transforms(size):
    extra    = {'additional_targets': {'image2': 'image'}}
    train_tf = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.03, scale=(0.93,1.07),
                 rotate=(-10,10), p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.1, hue=0.03, p=0.4),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.CoarseDropout(num_holes_range=(1,2),
                        hole_height_range=(16,32),
                        hole_width_range=(16,32),
                        fill=128, p=0.15),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ], **extra)
    val_tf = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ], **extra)
    return train_tf, val_tf


# ================= 类别权重（只用sqrt，不手动干预）=================
def get_class_weights(ds, num_classes, device):
    c   = Counter(ds.labels)
    tot = len(ds.labels)
    w   = [(tot / c.get(i,1)) ** 0.5 for i in range(num_classes)]
    mw  = sum(w) / len(w)
    w   = [x/mw for x in w]
    wt  = torch.tensor(w, dtype=torch.float32).to(device)
    cnames = ['正常','轻度','中度','重度','增殖期']
    logger.info("类别权重(sqrt自动):")
    for i, fw in enumerate(w):
        logger.info(f"  {cnames[i]}(类别{i}) n={c.get(i,0):>3}: w={fw:.3f}")
    return wt


# ================= 早停 =================
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.triggered = False

    def step(self, v):
        if self.best is None or v > self.best + self.min_delta:
            self.best = v; self.counter = 0
        else:
            self.counter += 1
            logger.info(f"ES {self.counter}/{self.patience} best={self.best:.4f}")
            if self.counter >= self.patience: self.triggered = True
        return self.triggered


# ================= 绘图 =================
def save_plots(h, out):
    ep = range(1, len(h['train_loss'])+1)
    plt.figure(figsize=(14,10))
    for i,(k1,k2,t) in enumerate([
        ('train_loss','val_loss','Loss'),
        ('train_qwk','val_qwk','QWK'),
        ('train_acc','val_acc','Accuracy')
    ]):
        plt.subplot(2,2,i+1)
        plt.plot(ep,h[k1],label=k1,marker='o',ms=3)
        plt.plot(ep,h[k2],label=k2,marker='o',ms=3)
        plt.title(t); plt.xlabel('Epochs')
        plt.legend(); plt.grid(True)
    plt.subplot(2,2,4)
    plt.plot(ep,h['lr'],color='orange',marker='o',ms=3,label='LR')
    plt.title('LR'); plt.xlabel('Epochs')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out,'training_history.png')); plt.close()


# ================= 主训练 =================
def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("=== 构建缓存 ===")
    build_cache(cfg.train_csv, cfg.train_img_dir, cfg.cache_train, 512)
    build_cache(cfg.val_csv,   cfg.val_img_dir,   cfg.cache_val,   512)

    logger.info("=== 少数类增强 ===")
    aug_records = build_minority_augmentation(
        cfg.train_csv, cfg.cache_train, cfg.aug_cache,
        cfg.minority_target, 512)

    train_tf, val_tf = get_transforms(512)
    train_ds = IDRiDDataset(cfg.train_csv, cfg.cache_train,
                            transform=train_tf,
                            aug_dir=cfg.aug_cache,
                            aug_records=aug_records)
    val_ds   = IDRiDDataset(cfg.val_csv, cfg.cache_val, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = FundusClassifier(cfg.num_classes).to(cfg.device)
    cls_w     = get_class_weights(train_ds, cfg.num_classes, cfg.device)
    criterion = FocalLoss(alpha=cls_w, gamma=2.0)

    # ===== 关键：一次性建好optimizer，包含两个参数组 =====
    # 冻结期：backbone requires_grad=False，梯度不传播但scheduler正常计数
    # 解冻后：backbone requires_grad=True，自然开始更新
    model.set_backbone_grad(False)

    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(),
         'lr': cfg.max_lr_head},
        {'params': model.features.parameters(),
         'lr': cfg.max_lr_bb},          # 冻结期此组不实际更新
    ], weight_decay=cfg.weight_decay)

    # ===== 全程单一OneCycleLR，不重建 =====
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = [cfg.max_lr_head, cfg.max_lr_bb],
        steps_per_epoch  = len(train_loader),
        epochs           = cfg.epochs,
        pct_start        = 0.1,
        anneal_strategy  = 'cos',
        div_factor       = 25.0,
        final_div_factor = 1e4,
    )

    scaler   = GradScaler('cuda')
    es       = EarlyStopping(cfg.early_stop_patience, cfg.min_delta)
    best_qwk = -1.0
    unfrozen = False

    history = {k: [] for k in
               ['train_loss','val_loss','train_qwk','val_qwk',
                'train_acc','val_acc','lr']}

    logger.info(f"设备:{cfg.device} | B0 6ch | 训练:{len(train_ds)} | 验证:{len(val_ds)}")
    logger.info(f"全程单一OneCycleLR，freeze={cfg.freeze_epochs}轮后解冻backbone")

    for epoch in range(1, cfg.epochs+1):

        # 解冻backbone（只改requires_grad，不动optimizer/scheduler）
        if epoch == cfg.freeze_epochs + 1 and not unfrozen:
            model.set_backbone_grad(True)
            unfrozen = True
            logger.info(f"Epoch{epoch}: 解冻backbone，LR平滑继续")

        model.train()
        tl, tp, tt = 0.0, [], []
        for imgs, labels in tqdm(train_loader,
                                 desc=f"Ep{epoch:>3}/{cfg.epochs} [Train]"):
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            with autocast('cuda'):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()      # OneCycleLR是step级，始终在batch循环内
            tl += loss.item() * imgs.size(0)
            _, pred = torch.max(out, 1)
            tp.extend(pred.cpu().numpy())
            tt.extend(labels.cpu().numpy())

        et_loss = tl / len(train_ds)
        et_qwk  = cohen_kappa_score(tt, tp, weights='quadratic')
        et_acc  = np.mean(np.array(tp) == np.array(tt))

        model.eval()
        vl, vp, vt = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                     desc=f"Ep{epoch:>3}/{cfg.epochs} [Val]  "):
                imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
                with autocast('cuda'):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                vl += loss.item() * imgs.size(0)
                _, pred = torch.max(out, 1)
                vp.extend(pred.cpu().numpy())
                vt.extend(labels.cpu().numpy())

        ev_loss = vl / len(val_ds)
        ev_qwk  = cohen_kappa_score(vt, vp, weights='quadratic')
        ev_acc  = np.mean(np.array(vp) == np.array(vt))
        cur_lr  = scheduler.get_last_lr()[0]

        mild_c    = sum(1 for t,p in zip(vt,vp) if t==1 and p==1)
        mild_t    = sum(1 for t in vt if t==1)
        mild_pred = Counter(p for t,p in zip(vt,vp) if t==1)

        logger.info(
            f"Ep{epoch:>3} unfrz={unfrozen} lr={cur_lr:.2e} | "
            f"T qwk={et_qwk:.4f} acc={et_acc:.4f} | "
            f"V qwk={ev_qwk:.4f} acc={ev_acc:.4f} loss={ev_loss:.4f} | "
            f"轻度:{mild_c}/{mild_t} pred={dict(mild_pred)}"
        )

        for k, v in zip(
            ['train_loss','val_loss','train_qwk','val_qwk','train_acc','val_acc','lr'],
            [et_loss,ev_loss,et_qwk,ev_qwk,et_acc,ev_acc,cur_lr]
        ):
            history[k].append(v)

        save_plots(history, cfg.output_dir)

        if ev_qwk > best_qwk:
            best_qwk = ev_qwk
            torch.save(model.state_dict(),
                       os.path.join(cfg.output_dir, "best_cls_model_qwk.pth"))
            logger.info(f"  ★ 新最佳 Val QWK={ev_qwk:.4f}")

        if es.step(ev_qwk):
            logger.info(f"Early Stop! 最佳={es.best:.4f}")
            break

    logger.info(f"完成！最佳 Val QWK={best_qwk:.4f}")

#try
if __name__ == "__main__":
    main()