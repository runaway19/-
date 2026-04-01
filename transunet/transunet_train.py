    # --- START OF FILE transunet_train.py ---
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    import cv2
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import logging
    from tqdm import tqdm
    import datetime
    import json
    import random
    import warnings
    import torch.nn.functional as F
    import scipy.ndimage as ndimage
    import scipy.spatial as spatial

    # --- 导入你的 ViT 相关文件 ---
    from vit_seg_modeling import VisionTransformer, CONFIGS

    # 设置日志 (严格保持不变)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


    # 配置参数
    class Config:
        image_dir = "E:/DR_Classification/IDRiD_dataset/A._Segmentation/1._Original_Images/a._Training_Set"
        mask_dirs = [
            "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/a._Training_Set/1._Microaneurysms",
            "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/a._Training_Set/2._Haemorrhages",
            "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/a._Training_Set/3._Hard_Exudates",
            "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/a._Training_Set/4._Soft_Exudates"
        ]
        mask_suffixes = ["_MA", "_HE", "_EX", "_SE"]

        batch_size = 2
        learning_rate = 1e-4
        num_epochs = 200
        num_classes = 4
        input_channels = 3
        crop_size = 512

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)


    # --- 数据集类 ---
    class MedicalDataset(Dataset):
        def __init__(self, image_paths, mask_paths_list, transform=None, is_train=True):
            self.image_paths = image_paths
            self.mask_paths_list = mask_paths_list
            self.transform = transform
            self.is_train = is_train

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            merged = cv2.merge((l, a, b))
            image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

            h, w = image.shape[:2]
            masks = []
            for i, mask_paths in enumerate(self.mask_paths_list):
                mask_path = mask_paths[idx]
                if mask_path is None:
                    mask = np.zeros((h, w), dtype=np.float32)
                else:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 0).astype(float) if mask is not None else np.zeros((h, w), dtype=np.float32)
                masks.append(mask)

            if self.is_train:
                crop_h, crop_w = 512, 512
                pad_h = max(0, crop_h - h)
                pad_w = max(0, crop_w - w)
                if pad_h > 0 or pad_w > 0:
                    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    masks = [cv2.copyMakeBorder(m, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0) for m in masks]
                    h, w = image.shape[:2]

                # 【优化】概率降至0.75，给模型机会学习单纯的"黑色背景"，压制过度预测
                if random.random() < 0.75:
                    present_classes = [i for i, m in enumerate(masks) if np.sum(m) > 0]
                    target_mask = None
                    if len(present_classes) > 0:
                        weights = [4.0 if c == 0 else 3.0 if c == 3 else 1.0 for c in present_classes]
                        target_class_idx = random.choices(present_classes, weights=weights, k=1)[0]
                        target_mask = masks[target_class_idx]

                    if target_mask is not None:
                        lesion_indices = np.argwhere(target_mask > 0)
                        if len(lesion_indices) > 0:
                            center_idx = random.choice(lesion_indices)
                            top = max(0, min(center_idx[0] - crop_h // 2 + random.randint(-30, 30), h - crop_h))
                            left = max(0, min(center_idx[1] - crop_w // 2 + random.randint(-30, 30), w - crop_w))
                        else:
                            top, left = random.randint(0, h - crop_h), random.randint(0, w - crop_w)
                    else:
                        top, left = random.randint(0, h - crop_h), random.randint(0, w - crop_w)
                else:
                    top, left = random.randint(0, h - crop_h), random.randint(0, w - crop_w)

                image = image[top:top + crop_h, left:left + crop_w, :]
                masks = [m[top:top + crop_h, left:left + crop_w] for m in masks]

            if self.transform:
                data = {'image': image}
                for i, m in enumerate(masks): data[f'mask{i}'] = m
                augmented = self.transform(**data)
                image_tensor = augmented['image']
                mask_tensor = torch.stack([augmented[f'mask{i}'] for i in range(len(masks))], dim=0)
            else:
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
                mask_tensor = torch.from_numpy(np.stack(masks)).float()

            mask_tensor = (mask_tensor > 0.5).float()
            return image_tensor, mask_tensor


    # --- 数据增强处理 ---
    import albumentations as A
    from albumentations.pytorch import ToTensorV2


    def get_transforms(is_train=True):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if is_train:
            return A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(0.0, 0.2), rotate_limit=45, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Normalize(mean, std),
                ToTensorV2(),
            ], additional_targets={f'mask{i}': 'mask' for i in range(4)})
        else:
            return A.Compose([
                A.Normalize(mean, std),
                ToTensorV2(),
            ], additional_targets={f'mask{i}': 'mask' for i in range(4)})


    # =====================================================================
    # --- 【解决假阳性 & 增强概率锐度】的改良 Loss
    # =====================================================================

    class UltimateRobustLoss(nn.Module):
        def __init__(self, device, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.class_weights = torch.tensor([5.0, 1.0, 1.0, 2.0]).view(1, 4, 1, 1).to(device)

        def forward(self, logits, targets):
            # 1. 加权 Focal Loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            probs = torch.sigmoid(logits)

            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - p_t) ** self.gamma)
            focal_loss = (focal_weight * bce_loss * self.class_weights).mean()

            # 2.改良 Tversky
            # 【修改1】alpha_t 提高到 0.45，加重 FP(假阳性/过度预测) 的惩罚
            alpha_t = 0.45
            beta_t = 0.55
            # 【修改2】smooth 从 1.0 降到 1e-4，强迫模型吐出极度确信的概率值(要么0要么1)
            smooth = 1e-2

            probs_flat = probs.view(probs.size(0), probs.size(1), -1)
            targets_flat = targets.view(targets.size(0), targets.size(1), -1)

            TP = (probs_flat * targets_flat).sum(dim=2)
            FP = (probs_flat * (1 - targets_flat)).sum(dim=2)
            FN = ((1 - probs_flat) * targets_flat).sum(dim=2)

            tversky = (TP + smooth) / (TP + alpha_t * FP + beta_t * FN + smooth)
            tversky_loss = 1. - tversky.mean()

            return focal_loss * 10.0 + tversky_loss


    # --- 高效计算 HD95 辅助函数 ---
    def compute_hd95(pred, gt):
        """
        计算二值图像的 95% Hausdorff 距离。
        如果都没有目标，距离为0；如果一方有一方没有，给予极大惩罚值。
        """
        pred = pred.astype(bool)
        gt = gt.astype(bool)

        if np.all(pred == 0) and np.all(gt == 0):
            return 0.0
        if np.all(pred == 0) or np.all(gt == 0):
            return 512.0  # 图像的最大对角线级别惩罚

        # 提取边界
        pred_edges = pred ^ ndimage.binary_erosion(pred, structure=np.ones((3, 3)))
        gt_edges = gt ^ ndimage.binary_erosion(gt, structure=np.ones((3, 3)))

        pred_pts = np.argwhere(pred_edges)
        gt_pts = np.argwhere(gt_edges)

        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return 512.0

        # 构建 KD 树计算最近邻距离
        tree_pred = spatial.cKDTree(pred_pts)
        tree_gt = spatial.cKDTree(gt_pts)

        d1, _ = tree_gt.query(pred_pts)
        d2, _ = tree_pred.query(gt_pts)

        hd95 = np.percentile(np.concatenate([d1, d2]), 95)
        return hd95


    def calculate_metrics(pred, target, threshold=0.5):
        if not ((pred >= 0) & (pred <= 1)).all(): pred = torch.sigmoid(pred)
        pred_binary = (pred > threshold).float()
        tp = (pred_binary * target).sum(dim=(2, 3))
        fp = (pred_binary * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred_binary) * target).sum(dim=(2, 3))
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        return {'dice': dice.mean(dim=0)}


    def get_file_list(image_dir, mask_dirs, mask_suffixes):
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
        image_base_to_path = {os.path.splitext(f)[0]: os.path.join(image_dir, f) for f in image_files}
        mask_base_to_path_list = []
        for m_dir, suffix in zip(mask_dirs, mask_suffixes):
            m_files = os.listdir(m_dir)
            mapping = {}
            for f in m_files:
                base = os.path.splitext(f)[0]
                clean_base = base[:-len(suffix)] if base.endswith(suffix) else base
                mapping[clean_base] = os.path.join(m_dir, f)
            mask_base_to_path_list.append(mapping)

        image_paths, mask_paths_list = [], [[] for _ in range(len(mask_dirs))]
        for base_name in sorted(image_base_to_path.keys()):
            has_any = False
            for i, mapping in enumerate(mask_base_to_path_list):
                path = mapping.get(base_name)
                mask_paths_list[i].append(path)
                if path: has_any = True
            if has_any:
                image_paths.append(image_base_to_path[base_name])
            else:
                for i in range(len(mask_dirs)): mask_paths_list[i].pop()
        return image_paths, mask_paths_list


    def predict_sliding_window(model, image_tensor, window_size=512, stride=256, num_classes=4):
        model.eval()
        b, c, h, w = image_tensor.shape
        device = image_tensor.device

        def get_gaussian(size, sigma_scale=1.0 / 8):
            tmp = torch.arange(size, device=device) - size // 2
            sigma = size * sigma_scale
            gauss1d = torch.exp(-tmp ** 2 / (2 * sigma ** 2))
            gauss2d = torch.outer(gauss1d, gauss1d)
            return gauss2d / gauss2d.max()

        weight_map = get_gaussian(window_size).unsqueeze(0).unsqueeze(0)
        full_probs = torch.zeros((b, num_classes, h, w), device=device)
        full_weights = torch.zeros((b, num_classes, h, w), device=device)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1, x1 = y, x
                y2, x2 = min(y + window_size, h), min(x + window_size, w)
                if y2 - y1 < window_size: y1 = max(0, h - window_size); y2 = h
                if x2 - x1 < window_size: x1 = max(0, w - window_size); x2 = w

                img_crop = image_tensor[:, :, y1:y2, x1:x2]
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred_crop = torch.sigmoid(model(img_crop))

                full_probs[:, :, y1:y2, x1:x2] += pred_crop.float() * weight_map
                full_weights[:, :, y1:y2, x1:x2] += weight_map

        return full_probs / (full_weights + 1e-7)


    # ====================================================
    # 训练主函数
    # ====================================================
    def train_model():
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 50}\n开始时间: {start_time_str}\n{'=' * 50}\n")

        config = Config()
        device = config.device
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"模型适配尺寸: {config.crop_size}...")
        vit_config = CONFIGS['R50-ViT-B_16']
        vit_config.n_classes = config.num_classes
        vit_config.patches.grid = (config.crop_size // 16, config.crop_size // 16)

        # 初始化模型
        model = VisionTransformer(vit_config, img_size=config.crop_size, num_classes=config.num_classes).to(device)

        # 加载预训练权重 (Windows 兼容)
        pretrained_path = vit_config.pretrained_path
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"正在加载官方预训练权重: {pretrained_path}")
            raw_weights = np.load(pretrained_path)

            class SlashAgnosticDict:
                def __init__(self, raw_dict):
                    self.raw_dict = raw_dict

                def __getitem__(self, key):
                    return self.raw_dict[key.replace('\\', '/')]

                def __contains__(self, key):
                    return key.replace('\\', '/') in self.raw_dict

                def keys(self):
                    return self.raw_dict.keys()

            adapted_weights = SlashAgnosticDict(raw_weights)
            model.load_from(weights=adapted_weights)
            logger.info("加载官方预训练权重")
        else:
            logger.error(f"未找到预训练权重 {pretrained_path}，模型将随机初始化！")

        # 准备数据
        image_paths, mask_paths_list = get_file_list(config.image_dir, config.mask_dirs, config.mask_suffixes)
        if len(image_paths) == 0: return

        samples = []
        for i in range(len(image_paths)):
            sample_masks = [mask_paths[i] for mask_paths in mask_paths_list]
            samples.append((image_paths[i], sample_masks))

        train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

        train_dataset = MedicalDataset([s[0] for s in train_samples],
                                       [[s[1][i] for s in train_samples] for i in range(4)],
                                       transform=get_transforms(is_train=True))
        val_dataset = MedicalDataset([s[0] for s in val_samples],
                                     [[s[1][i] for s in val_samples] for i in range(4)],
                                     transform=get_transforms(is_train=False), is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        criterion = UltimateRobustLoss(device=device)

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

        train_losses, val_losses = [], []
        val_metrics_history = []
        accumulation_steps = 4
        best_val_dice = 0.0

        logger.info("开始训练...")

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(config.num_epochs):
            model.train()
            epoch_train_loss = 0.0
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs} [Train]')

            optimizer.zero_grad()

            for batch_idx, (images, masks) in enumerate(train_bar):
                images, masks = images.to(device), masks.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    total_loss = criterion(outputs, masks)

                scaler.scale(total_loss / accumulation_steps).backward()

                is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))

                if is_update_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_train_loss += total_loss.item()

                with torch.no_grad():
                    metrics = calculate_metrics(outputs, masks, threshold=0.5)

                train_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Dice': f'{metrics["dice"].mean().item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

            scheduler.step()

            # --- 验证阶段 ---
            model.eval()
            total_tp = torch.zeros(4).to(device)
            total_fp = torch.zeros(4).to(device)
            total_fn = torch.zeros(4).to(device)

            # 收集 HD95
            epoch_hd95_lists = [[] for _ in range(4)]

            val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1} [Val]')
            with torch.no_grad():
                for images, masks in val_bar:
                    images, masks = images.to(device), masks.to(device)

                    full_probs = predict_sliding_window(model, images, num_classes=4, stride=256)

                    # 将统一阈值改为0.5，检验真实概率锐度
                    thresholds = torch.tensor([0.4, 0.4, 0.4, 0.4]).to(device).view(1, 4, 1, 1)
                    preds = (full_probs > thresholds).float()

                    total_tp += (preds * masks).sum(dim=(0, 2, 3))
                    total_fp += (preds * (1 - masks)).sum(dim=(0, 2, 3))
                    total_fn += ((1 - preds) * masks).sum(dim=(0, 2, 3))

                    # 计算 HD95 (转到 CPU numpy 运算)
                    pred_np = preds.cpu().numpy()
                    mask_np = masks.cpu().numpy()

                    # batch_size 为 1
                    for b in range(pred_np.shape[0]):
                        for c in range(4):
                            hd = compute_hd95(pred_np[b, c], mask_np[b, c])
                            epoch_hd95_lists[c].append(hd)

            epoch_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-7)
            avg_val_dice = epoch_dice.mean().item()

            avg_hd95 = [np.mean(lst) for lst in epoch_hd95_lists]

            # 保持你的 logger 格式，补充 HD95 指标打印
            logger.info(
                f"Dice -> MA: {epoch_dice[0]:.4f}, HE: {epoch_dice[1]:.4f}, EX: {epoch_dice[2]:.4f}, SE: {epoch_dice[3]:.4f} | "
                f"HD95 -> MA: {avg_hd95[0]:.2f}, HE: {avg_hd95[1]:.2f}, EX: {avg_hd95[2]:.2f}, SE: {avg_hd95[3]:.2f}"
            )

            # --- 记录数据 ---
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(1.0 - avg_val_dice)

            # 保存带有 HD95 的指标历史
            val_metrics_history.append({
                'dice': epoch_dice.cpu().numpy().tolist(),
                'hd95': avg_hd95
            })

            # 保存最佳模型 (依然以 Dice 为主导标准，HD95起参考作用)
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), os.path.join(config.log_dir, f'best_model.pth'))

        # --- 保存日志与绘制曲线 ---
        def save_training_results(config, train_losses, val_losses, val_metrics_history, timestamp):
            summary = {
                'timestamp': timestamp,
                'config': {
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'num_epochs': config.num_epochs,
                    'crop_size': config.crop_size,
                },
                'results': {
                    'train_losses': [float(x) for x in train_losses],
                    'val_losses': [float(x) for x in val_losses],
                    'val_metrics': val_metrics_history,
                    'best_val_loss': float(min(val_losses)) if val_losses else None
                }
            }

            json_file = os.path.join(config.log_dir, f"training_summary_{timestamp}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)

            plt.figure(figsize=(10, 6))
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Val Loss (1 - Avg Dice)', linewidth=2)
            plt.title('Training and Validation Loss Curve', fontsize=16)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)

            plot_file = os.path.join(config.log_dir, f"loss_curve_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info("训练结束，正在生成日志和曲线图...")
        save_training_results(config, train_losses, val_losses, val_metrics_history, timestamp)

        torch.save({'model_state': model.state_dict(), 'val_metrics': val_metrics_history[-1]},
                   os.path.join(config.log_dir, f'vit_final_{timestamp}.pth'))
        logger.info("所有文件保存完成。")


    if __name__ == "__main__":
        train_model()