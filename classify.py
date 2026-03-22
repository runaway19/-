"""
架构简化：去掉MaskStream CNN，只保留面积统计特征

原来：图像流(1280) + MaskCNN(128) + 面积(32) = 1440维
现在：图像流(1280) + 面积统计(64) = 1344维

面积统计特征包括：
  - 4个mask的像素占比（MA/HE/EX/SE各占多少）
  - 这4个数字直接对应DR分级标准，比CNN学出来的更稳定
  - 不需要从头训练，不会出现数值爆炸

同时修复：
  - freeze期间backbone的参数组从optimizer中完全移除，避免scheduler异常
  - 解冻后重建optimizer，但用极小lr避免尖刺
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
import segmentation_models_pytorch as smp

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainConfig:
    train_img_dir  = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    val_img_dir    = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    train_csv      = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    val_csv        = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    output_dir     = "C:/Users/Administrator/Desktop/PythonProject/cls_output"
    cache_train    = "C:/Users/Administrator/Desktop/PythonProject/cache_train_v3"
    cache_val      = "C:/Users/Administrator/Desktop/PythonProject/cache_val_v3"
    aug_cache      = "C:/Users/Administrator/Desktop/PythonProject/cache_augmented_v5"
    seg_model_path = "C:/Users/Administrator/Desktop/PythonProject/final_model_20260102_140556.pth"

    seg_window     = 640
    seg_stride     = 480
    seg_thresholds = [0.4, 0.5, 0.5, 0.4]

    num_classes    = 5
    batch_size     = 16
    epochs         = 60

    # 阶段1：只训练分类头（backbone冻结）
    lr_head        = 5e-4
    # 阶段2：解冻后极小backbone lr
    lr_bb_finetune = 1e-5
    weight_decay   = 1e-2

    freeze_epochs  = 15   # 前15轮只训练分类头

    early_stop_patience = 20
    min_delta           = 1e-4
    minority_target     = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 分割模型 =================
def load_seg_model(model_path, device):
    model_ma = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", encoder_weights=None,
        in_channels=3, classes=1).to(device)
    model_rest = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", encoder_weights=None,
        in_channels=3, classes=3).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model_ma.load_state_dict(ckpt['model_ma_state'])
    model_rest.load_state_dict(ckpt['model_rest_state'])
    model_ma.eval(); model_rest.eval()
    for p in list(model_ma.parameters()) + list(model_rest.parameters()):
        p.requires_grad = False
    logger.info("分割模型加载成功")
    return model_ma, model_rest

def seg_sliding_window(model, t, ws, stride, nc, device):
    b, c, h, w = t.shape
    tmp  = torch.arange(ws, device=device) - ws//2
    g1d  = torch.exp(-tmp**2 / (2*(ws/8)**2))
    wmap = (torch.outer(g1d,g1d)/torch.outer(g1d,g1d).max()).unsqueeze(0).unsqueeze(0)
    fp   = torch.zeros((b, nc, h, w), device=device)
    fw   = torch.zeros((b, nc, h, w), device=device)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2 = min(y+ws, h); x2 = min(x+ws, w)
            if y2-y1 < ws: y1 = max(0, h-ws); y2 = h
            if x2-x1 < ws: x1 = max(0, w-ws); x2 = w
            with torch.no_grad():
                pred = torch.sigmoid(model(t[:,:,y1:y2,x1:x2]))
            fp[:,:,y1:y2,x1:x2] += pred * wmap
            fw[:,:,y1:y2,x1:x2] += wmap
    return fp / (fw + 1e-7)

def get_seg_masks(img_rgb, model_ma, model_rest, cfg):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8,8)).apply(l)
    img_c = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    tf = A.Compose([A.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225]), ToTensorV2()])
    t  = tf(image=img_c)['image'].unsqueeze(0).to(cfg.device)
    pm = seg_sliding_window(model_ma,   t, cfg.seg_window, cfg.seg_stride, 1, cfg.device)
    pr = seg_sliding_window(model_rest, t, cfg.seg_window, cfg.seg_stride, 3, cfg.device)
    fp = torch.cat([pm, pr], dim=1)[0]
    return np.stack([(fp[i].cpu().numpy() > cfg.seg_thresholds[i]).astype(np.float32)
                     for i in range(4)], axis=2)


# ================= 预处理与缓存 =================
def ben_graham(image_bgr, size):
    img  = cv2.resize(image_bgr, (size, size))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=size/30)
    enh  = cv2.addWeighted(img, 4, blur, -4, 128)
    mask = np.zeros_like(enh)
    cv2.circle(mask, (size//2,size//2), int(size*0.45), (1,1,1), -1)
    return (enh*mask + 128*(1-mask)).astype(np.uint8)

def apply_clahe(img_rgb):
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(3.0, (8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

def make_10ch(image_bgr, size, model_ma, model_rest, cfg):
    orig_rgb = cv2.cvtColor(cv2.resize(image_bgr,(size,size)), cv2.COLOR_BGR2RGB)
    enh_rgb  = apply_clahe(ben_graham(image_bgr, size))
    full_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks    = get_seg_masks(full_rgb, model_ma, model_rest, cfg)
    masks_r  = cv2.resize(masks, (size,size), interpolation=cv2.INTER_NEAREST)
    ch_img   = np.concatenate([orig_rgb, enh_rgb], axis=2).astype(np.float32)
    ch_mask  = (masks_r * 255).astype(np.float32)
    return np.concatenate([ch_img, ch_mask], axis=2).astype(np.float32)

def build_cache(csv_path, img_dir, cache_dir, size, model_ma, model_rest, cfg):
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    n  = 0
    for i in tqdm(range(len(df)), desc=f"缓存→{os.path.basename(cache_dir)}"):
        name = str(df.iloc[i,0])
        if not name.lower().endswith(('.jpg','.png','.jpeg','.tif')): name += '.jpg'
        dst = os.path.join(cache_dir, os.path.splitext(name)[0] + '.npy')
        if os.path.exists(dst): n+=1; continue
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None: continue
        np.save(dst, make_10ch(img, size, model_ma, model_rest, cfg))
        n += 1
    logger.info(f"缓存: {n}张 → {cache_dir}")


# ================= 少数类增强 =================
def build_minority_augmentation(csv_path, cache_dir, aug_dir, target, size):
    if os.path.exists(aug_dir): shutil.rmtree(aug_dir)
    os.makedirs(aug_dir, exist_ok=True)
    df        = pd.read_csv(csv_path)
    cls_files = {i: [] for i in range(5)}
    for i in range(len(df)):
        name  = os.path.splitext(str(df.iloc[i,0]))[0] + '.npy'
        label = int(df.iloc[i,1])
        path  = os.path.join(cache_dir, name)
        if os.path.exists(path): cls_files[label].append(path)

    aug_tf = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.9),
        A.Affine(translate_percent=0.05, scale=(0.85,1.15), rotate=(-30,30), p=0.8),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.7),
    ], additional_targets={'image2':'image'})

    records = []
    for cls_id, paths in cls_files.items():
        if len(paths) >= target: continue
        needed = target - len(paths)
        logger.info(f"类别{cls_id}: {len(paths)}张 → 增强{needed}张")
        for idx in range(needed):
            arr = np.load(random.choice(paths))
            ch1 = arr[:,:,:3].astype(np.uint8)
            ch2 = arr[:,:,3:6].astype(np.uint8)
            ch_mask = arr[:,:,6:]
            aug = aug_tf(image=ch1, image2=ch2)
            new_arr = np.concatenate([
                aug['image'].astype(np.float32),
                aug['image2'].astype(np.float32),
                ch_mask], axis=2)
            fname = f"aug_cls{cls_id}_{idx:04d}.npy"
            np.save(os.path.join(aug_dir, fname), new_arr)
            records.append((fname, cls_id))
    logger.info(f"增强完成: {len(records)}张")
    return records


# ================= Focal Loss =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma

    def forward(self, inputs, targets):
        ce  = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt  = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()


# ================= 简化模型：图像流 + 面积统计特征 =================
class FundusClassifierV2(nn.Module):
    """
    去掉MaskStream CNN，用面积统计特征替代：
    - 图像流：EfficientNet-B0处理6通道图像 → 1280维
    - 面积统计：4个mask的像素占比 → 经MLP嵌入 → 64维
    - 融合：1280+64 → 分类头

    优势：
    1. 无从头训练的大网络，完全避免数值爆炸
    2. 参数量极小，不会过拟合
    3. 面积占比直接对应DR分级标准（临床诊断依据）
    """
    def __init__(self, num_classes):
        super().__init__()

        # 图像流
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
        self.img_backbone   = base.features
        self.img_pool       = base.avgpool
        img_feat_dim        = base.classifier[1].in_features   # 1280

        # 面积统计特征嵌入（4维→64维）
        # 用两层MLP，让模型学习面积特征的非线性组合
        self.area_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )

        # 融合分类头
        fused_dim = img_feat_dim + 64   # 1280 + 64 = 1344
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(p=0.45),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img6, mask4):
        # 图像流
        img_feat  = self.img_backbone(img6)
        img_feat  = self.img_pool(img_feat)
        img_feat  = torch.flatten(img_feat, 1)         # (B, 1280)

        # 面积统计：每个mask的像素占比（0~1）
        area      = mask4.mean(dim=(2, 3))             # (B, 4)
        area_feat = self.area_embed(area)              # (B, 64)

        # 融合
        fused = torch.cat([img_feat, area_feat], dim=1)
        return self.classifier(fused)

    def get_head_params(self):
        """分类头 + area_embed的参数"""
        return list(self.area_embed.parameters()) + \
               list(self.classifier.parameters())

    def get_backbone_params(self):
        return list(self.img_backbone.parameters())

    def set_backbone_grad(self, requires_grad: bool):
        for p in self.img_backbone.parameters():
            p.requires_grad = requires_grad


# ================= Dataset =================
class IDRiDDataset(Dataset):
    def __init__(self, csv_path, cache_dir, img_transform=None,
                 aug_dir=None, aug_records=None):
        self.img_transform = img_transform
        df = pd.read_csv(csv_path)
        items, labels = [], []
        for i in range(len(df)):
            name = os.path.splitext(str(df.iloc[i,0]))[0] + '.npy'
            path = os.path.join(cache_dir, name)
            if os.path.exists(path):
                items.append(path); labels.append(int(df.iloc[i,1]))
        if aug_dir and aug_records:
            for (fname, lbl) in aug_records:
                items.append(os.path.join(aug_dir, fname)); labels.append(lbl)
        self.items = items; self.labels = labels
        c = Counter(labels)
        logger.info("训练集分布:")
        for k in sorted(c): logger.info(f"  类别{k}: {c[k]}张")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        arr     = np.load(self.items[idx])
        ch_img  = arr[:,:,:6].astype(np.uint8)
        ch_mask = arr[:,:,6:] / 255.0

        if self.img_transform:
            ch1 = ch_img[:,:,:3]; ch2 = ch_img[:,:,3:]
            aug = self.img_transform(image=ch1, image2=ch2)
            t_img = torch.cat([aug['image'], aug['image2']], dim=0)
        else:
            norm  = A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            t_img = torch.cat([
                torch.from_numpy(np.transpose(norm(image=ch_img[:,:,:3])['image'],(2,0,1))),
                torch.from_numpy(np.transpose(norm(image=ch_img[:,:,3:])['image'],(2,0,1))),
            ], dim=0)

        t_mask = torch.from_numpy(np.transpose(ch_mask, (2,0,1))).float()
        return t_img, t_mask, torch.tensor(self.labels[idx], dtype=torch.long)


# ================= 数据增强 =================
def get_transforms(size):
    extra    = {'additional_targets': {'image2': 'image'}}
    train_tf = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.03, scale=(0.93,1.07), rotate=(-10,10), p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03, p=0.4),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ], **extra)
    val_tf = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ], **extra)
    return train_tf, val_tf


# ================= 类别权重 =================
def get_class_weights(ds, num_classes, device):
    c   = Counter(ds.labels)
    tot = len(ds.labels)
    w   = [(tot/c.get(i,1))**0.5 for i in range(num_classes)]
    mw  = sum(w)/len(w)
    w   = [x/mw for x in w]
    wt  = torch.tensor(w, dtype=torch.float32).to(device)
    cnames = ['正常','轻度','中度','重度','增殖期']
    logger.info("类别权重:")
    for i,fw in enumerate(w): logger.info(f"  {cnames[i]}: n={c.get(i,0):>3} w={fw:.3f}")
    return wt


# ================= 早停 =================
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience=patience; self.min_delta=min_delta
        self.counter=0; self.best=None; self.triggered=False

    def step(self, v):
        if self.best is None or v > self.best + self.min_delta:
            self.best=v; self.counter=0
        else:
            self.counter+=1
            logger.info(f"ES {self.counter}/{self.patience} best={self.best:.4f}")
            if self.counter >= self.patience: self.triggered=True
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
        plt.title(t); plt.xlabel('Epochs'); plt.legend(); plt.grid(True)
    plt.subplot(2,2,4)
    plt.plot(ep,h['lr'],color='orange',marker='o',ms=3,label='LR')
    plt.title('LR'); plt.xlabel('Epochs'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out,'training_history.png')); plt.close()


# ================= 主训练 =================
def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("=== 加载分割模型 ===")
    model_ma, model_rest = load_seg_model(cfg.seg_model_path, cfg.device)

    logger.info("=== 构建缓存（已有则跳过）===")
    build_cache(cfg.train_csv, cfg.train_img_dir, cfg.cache_train,
                512, model_ma, model_rest, cfg)
    build_cache(cfg.val_csv,   cfg.val_img_dir,   cfg.cache_val,
                512, model_ma, model_rest, cfg)

    del model_ma, model_rest; torch.cuda.empty_cache()
    logger.info("分割模型已释放")

    logger.info("=== 少数类增强 ===")
    aug_records = build_minority_augmentation(
        cfg.train_csv, cfg.cache_train, cfg.aug_cache,
        cfg.minority_target, 512)

    train_tf, val_tf = get_transforms(512)
    train_ds = IDRiDDataset(cfg.train_csv, cfg.cache_train,
                            img_transform=train_tf,
                            aug_dir=cfg.aug_cache, aug_records=aug_records)
    val_ds   = IDRiDDataset(cfg.val_csv, cfg.cache_val, img_transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = FundusClassifierV2(cfg.num_classes).to(cfg.device)
    cls_w     = get_class_weights(train_ds, cfg.num_classes, cfg.device)
    criterion = FocalLoss(alpha=cls_w, gamma=2.0)

    # ===== 阶段1：冻结backbone，只优化head和area_embed =====
    model.set_backbone_grad(False)
    optimizer = optim.AdamW(
        model.get_head_params(),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg.lr_head,
        steps_per_epoch = len(train_loader),
        epochs          = cfg.epochs,
        pct_start       = 0.15,
        anneal_strategy = 'cos',
        div_factor      = 10.0,
        final_div_factor = 1e3,
    )

    scaler   = GradScaler('cuda')
    es       = EarlyStopping(cfg.early_stop_patience, cfg.min_delta)
    best_qwk = -1.0
    unfrozen = False

    history = {k: [] for k in
               ['train_loss','val_loss','train_qwk','val_qwk',
                'train_acc','val_acc','lr']}

    logger.info(f"设备:{cfg.device} | 简化双流(图像+面积统计) | "
                f"训练:{len(train_ds)} | 验证:{len(val_ds)}")
    logger.info(f"阶段1(前{cfg.freeze_epochs}轮): 只训练head+area_embed")
    logger.info(f"阶段2(第{cfg.freeze_epochs+1}轮起): 解冻backbone，极小lr微调")

    for epoch in range(1, cfg.epochs+1):

        # ===== 阶段2：解冻backbone，重建optimizer =====
        if epoch == cfg.freeze_epochs + 1 and not unfrozen:
            model.set_backbone_grad(True)
            # 分层lr：head用当前lr继续，backbone用极小lr
            optimizer = optim.AdamW([
                {'params': model.get_head_params(),     'lr': cfg.lr_head * 0.1},
                {'params': model.get_backbone_params(), 'lr': cfg.lr_bb_finetune},
            ], weight_decay=cfg.weight_decay)
            # 剩余轮数的CosineAnnealingLR（不用OneCycleLR避免重建问题）
            remaining = cfg.epochs - epoch + 1
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-7)
            unfrozen = True
            logger.info(f"Epoch{epoch}: 解冻backbone | "
                        f"head_lr={cfg.lr_head*0.1:.1e} bb_lr={cfg.lr_bb_finetune:.1e}")

        model.train()
        tl, tp, tt = 0.0, [], []
        for img6, mask4, labels in tqdm(train_loader,
                                        desc=f"Ep{epoch:>3}/{cfg.epochs} [Train]"):
            img6, mask4, labels = (img6.to(cfg.device), mask4.to(cfg.device),
                                   labels.to(cfg.device))
            optimizer.zero_grad()
            with autocast('cuda'):
                out  = model(img6, mask4)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            # 阶段1用step级scheduler，阶段2用epoch级
            if not unfrozen:
                scheduler.step()
            tl += loss.item() * img6.size(0)
            _, pred = torch.max(out, 1)
            tp.extend(pred.cpu().numpy()); tt.extend(labels.cpu().numpy())

        # 阶段2 epoch级更新
        if unfrozen:
            scheduler.step()

        et_loss = tl/len(train_ds)
        et_qwk  = cohen_kappa_score(tt, tp, weights='quadratic')
        et_acc  = np.mean(np.array(tp)==np.array(tt))

        model.eval()
        vl, vp, vt = 0.0, [], []
        with torch.no_grad():
            for img6, mask4, labels in tqdm(val_loader,
                                            desc=f"Ep{epoch:>3}/{cfg.epochs} [Val]  "):
                img6, mask4, labels = (img6.to(cfg.device), mask4.to(cfg.device),
                                       labels.to(cfg.device))
                with autocast('cuda'):
                    out  = model(img6, mask4)
                    loss = criterion(out, labels)
                vl += loss.item() * img6.size(0)
                _, pred = torch.max(out, 1)
                vp.extend(pred.cpu().numpy()); vt.extend(labels.cpu().numpy())

        ev_loss = vl/len(val_ds)
        ev_qwk  = cohen_kappa_score(vt, vp, weights='quadratic')
        ev_acc  = np.mean(np.array(vp)==np.array(vt))
        cur_lr  = optimizer.param_groups[0]['lr']

        mild_c    = sum(1 for t,p in zip(vt,vp) if t==1 and p==1)
        mild_pred = Counter(p for t,p in zip(vt,vp) if t==1)
        logger.info(
            f"Ep{epoch:>3} ph={'2' if unfrozen else '1'} lr={cur_lr:.2e} | "
            f"T qwk={et_qwk:.4f} acc={et_acc:.4f} | "
            f"V qwk={ev_qwk:.4f} acc={ev_acc:.4f} loss={ev_loss:.4f} | "
            f"轻度:{mild_c}/5 pred={dict(mild_pred)}"
        )

        for k,v in zip(
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


if __name__ == "__main__":
    main()