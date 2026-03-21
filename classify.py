"""
核心改动：将分割模型输出作为额外通道输入分类模型

原来输入：6通道（原图RGB + BenGraham增强RGB）
现在输入：10通道（原图RGB + BenGraham增强RGB + 4个分割mask）

4个分割mask对应：MA(微血管瘤) HE(出血) EX(硬性渗出) SE(软性渗出)
这些mask直接告诉分类模型"病变在哪里"，大幅提升判别力

分割模型只用于特征提取，推理时固定权重不参与训练
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
    train_img_dir = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    val_img_dir   = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    train_csv     = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    val_csv       = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    output_dir    = "C:/Users/Administrator/Desktop/PythonProject/cls_output"
    cache_train   = "C:/Users/Administrator/Desktop/PythonProject/cache_train_v3"   # 新缓存目录（10通道）
    cache_val     = "C:/Users/Administrator/Desktop/PythonProject/cache_val_v3"
    aug_cache     = "C:/Users/Administrator/Desktop/PythonProject/cache_augmented_v3"

    # 分割模型路径
    seg_model_path = "C:/Users/Administrator/Desktop/PythonProject/final_model_20260102_140556.pth"

    # 分割推理参数（与你的分割代码保持一致）
    seg_window    = 640
    seg_stride    = 480
    seg_thresholds = [0.4, 0.5, 0.5, 0.4]   # MA HE EX SE

    num_classes   = 5
    batch_size    = 16
    epochs        = 100

    max_lr_head   = 5e-4
    max_lr_bb     = 5e-5
    weight_decay  =2e-2
    freeze_epochs = 10

    early_stop_patience = 15
    min_delta           = 1e-4
    minority_target     = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 1. 分割模型加载 =================
def load_seg_model(model_path, device):
    """加载双模型分割结构，固定权重用于特征提取"""
    model_ma = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    model_rest = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=3,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model_ma.load_state_dict(checkpoint['model_ma_state'])
    model_rest.load_state_dict(checkpoint['model_rest_state'])

    model_ma.eval()
    model_rest.eval()

    # 固定权重，不参与训练
    for p in model_ma.parameters():
        p.requires_grad = False
    for p in model_rest.parameters():
        p.requires_grad = False

    logger.info(f"分割模型加载成功: {model_path}")
    return model_ma, model_rest


# ================= 2. 分割推理（滑动窗口）=================
def seg_sliding_window(model, image_tensor, window_size, stride,
                        num_classes, device):
    """与你的分割推理代码逻辑完全一致"""
    b, c, h, w = image_tensor.shape

    def get_gaussian(ws):
        tmp   = torch.arange(ws, device=device) - ws // 2
        sigma = ws / 8.0
        g1d   = torch.exp(-tmp**2 / (2 * sigma**2))
        g2d   = torch.outer(g1d, g1d)
        return (g2d / g2d.max()).unsqueeze(0).unsqueeze(0)

    weight_map   = get_gaussian(window_size)
    full_probs   = torch.zeros((b, num_classes, h, w), device=device)
    full_weights = torch.zeros((b, num_classes, h, w), device=device)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2 = min(y + window_size, h)
            x2 = min(x + window_size, w)
            if y2 - y1 < window_size: y1 = max(0, h - window_size); y2 = h
            if x2 - x1 < window_size: x1 = max(0, w - window_size); x2 = w

            crop = image_tensor[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                pred = torch.sigmoid(model(crop))
            full_probs[:, :, y1:y2, x1:x2]   += pred * weight_map
            full_weights[:, :, y1:y2, x1:x2] += weight_map

    return full_probs / (full_weights + 1e-7)


def get_seg_masks(image_rgb_np, model_ma, model_rest, cfg):
    """
    输入：RGB numpy图像 (H,W,3)
    输出：4通道mask numpy (H,W,4)，值为0或1的float
    """
    # 预处理（与分割训练保持一致）
    lab = cv2.cvtColor(image_rgb_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    img_clahe = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

    tf = A.Compose([
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    t = tf(image=img_clahe)['image'].unsqueeze(0).to(cfg.device)

    pred_ma   = seg_sliding_window(model_ma,   t, cfg.seg_window,
                                    cfg.seg_stride, 1, cfg.device)
    pred_rest = seg_sliding_window(model_rest, t, cfg.seg_window,
                                    cfg.seg_stride, 3, cfg.device)
    full_pred = torch.cat([pred_ma, pred_rest], dim=1)[0]   # (4,H,W)

    # 二值化
    masks = []
    for i, thr in enumerate(cfg.seg_thresholds):
        masks.append((full_pred[i].cpu().numpy() > thr).astype(np.float32))

    return np.stack(masks, axis=2)   # (H,W,4)


# ================= 3. 预处理与10通道缓存 =================
def ben_graham(image_bgr, size):
    img  = cv2.resize(image_bgr, (size, size))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=size/30)
    enh  = cv2.addWeighted(img, 4, blur, -4, 128)
    mask = np.zeros_like(enh)
    cv2.circle(mask, (size//2,size//2), int(size*0.45), (1,1,1), -1)
    return (enh * mask + 128*(1-mask)).astype(np.uint8)

def apply_clahe(img_rgb):
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(3.0,(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

def make_10ch(image_bgr, size, model_ma, model_rest, cfg):
    """
    生成10通道图像：
    通道 0-2:  原图 RGB
    通道 3-5:  Ben Graham增强 RGB
    通道 6-9:  分割mask（MA HE EX SE），float32
    """
    orig_rgb = cv2.cvtColor(cv2.resize(image_bgr,(size,size)), cv2.COLOR_BGR2RGB)
    enh_rgb  = apply_clahe(ben_graham(image_bgr, size))

    # 分割mask（在原始尺寸推理，然后resize到size）
    orig_rgb_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = get_seg_masks(orig_rgb_full, model_ma, model_rest, cfg)  # (H,W,4)
    masks_resized = cv2.resize(masks, (size, size),
                                interpolation=cv2.INTER_NEAREST)      # (size,size,4)

    # 合并：uint8部分 + float32 mask
    ch_img  = np.concatenate([orig_rgb, enh_rgb], axis=2).astype(np.float32)  # (s,s,6)
    ch_mask = (masks_resized * 255).astype(np.float32)                          # (s,s,4)
    result  = np.concatenate([ch_img, ch_mask], axis=2)                         # (s,s,10)
    return result.astype(np.float32)

def build_cache(csv_path, img_dir, cache_dir, size, model_ma, model_rest, cfg):
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
        arr = make_10ch(img, size, model_ma, model_rest, cfg)
        np.save(dst, arr)
        n += 1
    logger.info(f"缓存: {n}张 → {cache_dir}")


# ================= 4. 少数类增强 =================
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

    # 几何增强同步应用到所有10通道
    aug_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.9),
        A.Affine(translate_percent=0.05, scale=(0.85,1.15),
                 rotate=(-30,30), p=0.8),
    ], additional_targets={f'image{i}': 'image' for i in range(1, 10)})

    records = []
    for cls_id, paths in cls_files.items():
        if len(paths) >= target: continue
        needed = target - len(paths)
        logger.info(f"类别{cls_id}: {len(paths)}张 → 增强{needed}张")
        for idx in range(needed):
            arr = np.load(random.choice(paths))   # (H,W,10) float32

            # 颜色增强只用于前6通道（图像通道），mask通道用几何增强
            # 分离各通道
            chs = [arr[:,:,i].astype(np.uint8) for i in range(6)]
            mks = [arr[:,:,6+i] for i in range(4)]

            # 颜色增强
            color_aug = A.Compose([
                A.ColorJitter(brightness=0.35, contrast=0.35,
                              saturation=0.25, hue=0.06, p=0.8),
                A.RandomGamma(gamma_limit=(70,130), p=0.4),
                A.GaussianBlur(blur_limit=(3,7), p=0.3),
            ])
            ch0 = color_aug(image=np.stack(chs[:3],axis=2))['image']
            ch1 = color_aug(image=np.stack(chs[3:],axis=2))['image']

            # 几何增强（所有通道同步）
            geom_aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.9),
                A.Affine(translate_percent=0.05, scale=(0.85,1.15),
                         rotate=(-30,30), p=0.8),
            ], additional_targets={'image1':'image','image2':'image',
                                   'image3':'image','image4':'image'})

            res = geom_aug(
                image   = ch0,
                image1  = ch1,
                image2  = (mks[0]*255).astype(np.uint8)[:,:,np.newaxis].repeat(3,axis=2),
                image3  = (mks[1]*255).astype(np.uint8)[:,:,np.newaxis].repeat(3,axis=2),
                image4  = (mks[2]*255).astype(np.uint8)[:,:,np.newaxis].repeat(3,axis=2),
            )

            new_arr = np.concatenate([
                res['image'].astype(np.float32),
                res['image1'].astype(np.float32),
                (res['image2'][:,:,:1]).astype(np.float32),
                (res['image3'][:,:,:1]).astype(np.float32),
                (res['image4'][:,:,:1]).astype(np.float32),
                (mks[3]*255).astype(np.float32)[:,:,np.newaxis],   # SE mask
            ], axis=2)   # (H,W,10)

            fname = f"aug_cls{cls_id}_{idx:04d}.npy"
            np.save(os.path.join(aug_dir, fname), new_arr)
            records.append((fname, cls_id))

    logger.info(f"增强完成: {len(records)}张")
    return records


# ================= 5. Focal Loss =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce  = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ================= 6. 模型：EfficientNet-B0，10通道输入 =================
class FundusClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base     = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        old_conv = base.features[0][0]

        # 10通道输入
        new_conv = nn.Conv2d(
            10, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding, bias=False)
        with torch.no_grad():
            # 前3通道：原图，用预训练权重
            new_conv.weight[:, :3]  = old_conv.weight.clone()
            # 通道3-5：Ben Graham增强，复制预训练权重
            new_conv.weight[:, 3:6] = old_conv.weight.clone()
            # 通道6-9：分割mask，用小随机值初始化
            nn.init.kaiming_normal_(new_conv.weight[:, 6:])
            new_conv.weight[:, 6:] *= 0.1   # 初始时mask贡献小，让模型慢慢学

        base.features[0][0] = new_conv

        num_features  = base.classifier[1].in_features  # 1280
        self.features = base.features
        self.avgpool  = base.avgpool

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.5),
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


# ================= 7. Dataset =================
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
        arr = np.load(self.items[idx])   # (H,W,10) float32

        # 前6通道是uint8范围，后4通道是0/255 float
        ch_img  = arr[:,:,:6].astype(np.uint8)    # (H,W,6)
        ch_mask = arr[:,:,6:] / 255.0              # (H,W,4) 归一化到0-1

        if self.transform:
            # 图像通道做增强，mask通道做同步几何增强
            ch1 = ch_img[:,:,:3]
            ch2 = ch_img[:,:,3:]
            aug = self.transform(image=ch1, image2=ch2)
            t_img = torch.cat([aug['image'], aug['image2']], dim=0)  # (6,H,W)
        else:
            norm = A.Normalize(mean=[0.485,0.456,0.406],
                               std=[0.229,0.224,0.225])
            ch1_n = norm(image=ch_img[:,:,:3])['image']
            ch2_n = norm(image=ch_img[:,:,3:])['image']
            t_img = torch.from_numpy(
                np.transpose(np.concatenate([ch1_n, ch2_n], axis=2), (2,0,1)))

        # mask通道转tensor
        t_mask = torch.from_numpy(
            np.transpose(ch_mask, (2,0,1))).float()  # (4,H,W)

        t = torch.cat([t_img, t_mask], dim=0)  # (10,H,W)
        return t, torch.tensor(self.labels[idx], dtype=torch.long)


# ================= 8. 数据增强 =================
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


# ================= 9. 类别权重 =================
def get_class_weights(ds, num_classes, device):
    c   = Counter(ds.labels)
    tot = len(ds.labels)
    w   = [(tot / c.get(i,1)) ** 0.5 for i in range(num_classes)]
    mw  = sum(w) / len(w)
    w   = [x/mw for x in w]
    wt  = torch.tensor(w, dtype=torch.float32).to(device)
    cnames = ['正常','轻度','中度','重度','增殖期']
    logger.info("类别权重:")
    for i, fw in enumerate(w):
        logger.info(f"  {cnames[i]}: n={c.get(i,0):>3} w={fw:.3f}")
    return wt


# ================= 10. 早停 =================
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


# ================= 11. 绘图 =================
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


# ================= 12. 主训练 =================
def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 加载分割模型（只用于缓存构建，不参与训练梯度）
    logger.info("=== 加载分割模型 ===")
    model_ma, model_rest = load_seg_model(cfg.seg_model_path, cfg.device)

    logger.info("=== 构建10通道缓存（含分割mask）===")
    build_cache(cfg.train_csv, cfg.train_img_dir, cfg.cache_train,
                512, model_ma, model_rest, cfg)
    build_cache(cfg.val_csv,   cfg.val_img_dir,   cfg.cache_val,
                512, model_ma, model_rest, cfg)

    # 缓存建好后释放分割模型显存
    del model_ma, model_rest
    torch.cuda.empty_cache()
    logger.info("分割模型已释放，开始分类训练")

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

    model.set_backbone_grad(False)
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': cfg.max_lr_head},
        {'params': model.features.parameters(),   'lr': cfg.max_lr_bb},
    ], weight_decay=cfg.weight_decay)

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

    logger.info(f"设备:{cfg.device} | B0 10ch(6图像+4mask) | "
                f"训练:{len(train_ds)} | 验证:{len(val_ds)}")

    for epoch in range(1, cfg.epochs+1):

        if epoch == cfg.freeze_epochs + 1 and not unfrozen:
            model.set_backbone_grad(True)
            unfrozen = True
            logger.info(f"Epoch{epoch}: 解冻backbone")

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
            scheduler.step()
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
        mild_pred = Counter(p for t,p in zip(vt,vp) if t==1)

        logger.info(
            f"Ep{epoch:>3} unfrz={unfrozen} lr={cur_lr:.2e} | "
            f"T qwk={et_qwk:.4f} acc={et_acc:.4f} | "
            f"V qwk={ev_qwk:.4f} acc={ev_acc:.4f} loss={ev_loss:.4f} | "
            f"轻度:{mild_c}/5 pred={dict(mild_pred)}"
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


if __name__ == "__main__":
    main()