# --- START OF FILE transunet_test.py ---
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import json
import random

# --- 导入 TransUNet 相关文件 ---
from vit_seg_modeling import VisionTransformer, CONFIGS

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'


class TestConfig:
    # 测试数据路径 - 请确认路径正确
    test_image_dir = "E:/DR_Classification/IDRiD_dataset/A._Segmentation/1._Original_Images/b._Testing_Set"
    test_mask_dirs = [
        "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/b._Testing_Set/1._Microaneurysms",
        "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/b._Testing_Set/2._Haemorrhages",
        "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/b._Testing_Set/3._Hard_Exudates",
        "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/b._Testing_Set/4._Soft_Exudates"
    ]

    mask_suffixes = ["_MA", "_HE", "_EX", "_SE"]

    # 模型参数
    batch_size = 1  # 推理建议 batch_size = 1
    num_classes = 4
    crop_size = 512  # 必须和训练时一致
    stride = 256  # 滑窗推断步长，256 保证精度最高

    # 类别判定阈值：MA 极难，阈值放宽至 0.2
    thresholds = [0.4, 0.5, 0.4, 0.4]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 结果保存路径
    output_dir = "./outputs/transunet_test_result"
    os.makedirs(output_dir, exist_ok=True)


class TestDataset(Dataset):
    def __init__(self, image_paths, mask_paths_list, transform=None):
        self.image_paths = image_paths
        self.mask_paths_list = mask_paths_list
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        # ==========================================================
        # 【致命错误修复】必须保持和训练时一模一样的 CLAHE 预处理！
        # ==========================================================
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        original_height, original_width = image.shape[:2]

        # 读取掩码
        masks = []
        for i, mask_paths in enumerate(self.mask_paths_list):
            mask_path = mask_paths[idx]
            if mask_path is None:
                mask = np.zeros((original_height, original_width), dtype=np.float32)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((original_height, original_width), dtype=np.float32)
                else:
                    mask = (mask > 0).astype(float)
            masks.append(mask)

        # 应用变换
        if self.transform:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            additional_targets = {f'mask{i}': 'mask' for i in range(len(masks))}
            to_transform = {'image': image}
            for i, mask in enumerate(masks):
                to_transform[f'mask{i}'] = mask

            augmented = self.transform(**to_transform)
            image_tensor = augmented['image']
            mask_tensor = torch.stack([augmented[f'mask{i}'] for i in range(len(masks))], dim=0)
        else:
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask_tensor = torch.from_numpy(np.stack(masks)).float()

        return image_tensor, mask_tensor, image_path


def get_test_transforms():
    """测试时的数据变换 (仅归一化)"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={f'mask{i}': 'mask' for i in range(4)})


def get_file_list(image_dir, mask_dirs, mask_suffixes):
    """复用老代码中的鲁棒文件匹配逻辑"""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
    logger.info(f"找到 {len(image_files)} 个测试图像文件")

    image_base_to_path = {os.path.splitext(f)[0]: os.path.join(image_dir, f) for f in image_files}
    mask_base_to_path_list = []

    for i, (mask_dir, suffix) in enumerate(zip(mask_dirs, mask_suffixes)):
        mask_base_to_path = {}
        if os.path.exists(mask_dir):
            mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]
            for mask_file in mask_files:
                base_name = os.path.splitext(mask_file)[0]
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                mask_base_to_path[base_name] = os.path.join(mask_dir, mask_file)
        mask_base_to_path_list.append(mask_base_to_path)

    image_paths = []
    mask_paths_list = [[] for _ in range(len(mask_dirs))]
    class_counts = [0] * len(mask_dirs)

    for base_name in sorted(image_base_to_path.keys()):
        has_at_least_one_mask = False
        for i, mask_base_to_path in enumerate(mask_base_to_path_list):
            if base_name in mask_base_to_path:
                mask_paths_list[i].append(mask_base_to_path[base_name])
                class_counts[i] += 1
                has_at_least_one_mask = True
            else:
                mask_paths_list[i].append(None)

        if has_at_least_one_mask:
            image_paths.append(image_base_to_path[base_name])
        else:
            for i in range(len(mask_dirs)): mask_paths_list[i].pop()

    for i, count in enumerate(class_counts):
        logger.info(f"类别 {i} ({mask_suffixes[i]}): {count} 个 GT mask")

    return image_paths, mask_paths_list


# ==========================================
# 可视化与指标计算
# ==========================================
def create_comparison_visualization(image, true_masks, pred_masks, save_path, class_names, thresholds):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_vis = image.astype(np.uint8).copy()

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 12)

    ax1 = fig.add_subplot(gs[0, 0:4])
    ax1.imshow(image_vis)
    ax1.set_title("Original Image", fontsize=12, weight='bold')
    ax1.axis('off')

    class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    gt_overlay = image_vis.copy()
    gt_legend = []
    for i, color in enumerate(class_colors):
        mask = (true_masks[i] > 0.5).astype(bool)
        if mask.sum() > 0:
            colored_mask = np.zeros_like(image_vis)
            colored_mask[mask] = color
            mask_indices = np.any(colored_mask > 0, axis=-1)
            gt_overlay[mask_indices] = cv2.addWeighted(image_vis[mask_indices], 0.6, colored_mask[mask_indices], 0.4, 0)
            gt_legend.append(f"{class_names[i]}")

    ax2 = fig.add_subplot(gs[0, 4:8])
    ax2.imshow(gt_overlay)
    ax2.set_title(f"Ground Truth (Merged)\n{', '.join(gt_legend)}", fontsize=12, weight='bold')
    ax2.axis('off')

    pred_overlay = image_vis.copy()
    for i, color in enumerate(class_colors):
        mask = (pred_masks[i] > thresholds[i]).astype(bool)
        if mask.sum() > 0:
            colored_mask = np.zeros_like(image_vis)
            colored_mask[mask] = color
            mask_indices = np.any(colored_mask > 0, axis=-1)
            pred_overlay[mask_indices] = cv2.addWeighted(image_vis[mask_indices], 0.6, colored_mask[mask_indices], 0.4,
                                                         0)

    ax3 = fig.add_subplot(gs[0, 8:12])
    ax3.imshow(pred_overlay)
    ax3.set_title(f"Prediction (Merged)", fontsize=12, weight='bold')
    ax3.axis('off')

    for i in range(4):
        col_start, col_end = i * 3, (i + 1) * 3
        ax = fig.add_subplot(gs[1, col_start:col_end])

        t_mask = (true_masks[i] > 0.5).astype(bool)
        p_mask = (pred_masks[i] > thresholds[i]).astype(bool)

        tp, fp, fn = t_mask & p_mask, (~t_mask) & p_mask, t_mask & (~p_mask)

        error_overlay = image_vis.copy()
        error_overlay[tp] = [0, 255, 0]  # Green: Correct
        error_overlay[fp] = [255, 0, 0]  # Red: False Positive
        error_overlay[fn] = [0, 0, 255]  # Blue: False Negative

        intersection, union = tp.sum(), t_mask.sum() + p_mask.sum()
        dice = (2.0 * intersection) / union if union > 0 else 0.0

        ax.imshow(error_overlay)
        ax.set_title(f"{class_names[i]} (T={thresholds[i]})\nDice: {dice:.3f}", fontsize=10, weight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_iou(true_mask, pred_mask):
    intersection, union = np.logical_and(true_mask, pred_mask).sum(), np.logical_or(true_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0


def calculate_dice(true_mask, pred_mask):
    intersection, total = np.logical_and(true_mask, pred_mask).sum(), true_mask.sum() + pred_mask.sum()
    return (2.0 * intersection) / total if total > 0 else 0.0


def calculate_precision(true_mask, pred_mask):
    tp, fp = np.logical_and(true_mask, pred_mask).sum(), np.logical_and(~true_mask, pred_mask).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_recall(true_mask, pred_mask):
    tp, fn = np.logical_and(true_mask, pred_mask).sum(), np.logical_and(true_mask, ~pred_mask).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def save_results_to_json(all_metrics, class_names, output_path):
    results = {'overall_metrics': {}, 'class_metrics': {}}
    for i, class_name in enumerate(class_names):
        results['class_metrics'][class_name] = {
            'iou': float(all_metrics['iou'][i]), 'dice': float(all_metrics['dice'][i]),
            'precision': float(all_metrics['precision'][i]), 'recall': float(all_metrics['recall'][i])
        }
    results['overall_metrics'] = {
        'mean_iou': float(np.mean(all_metrics['iou'])), 'mean_dice': float(np.mean(all_metrics['dice'])),
        'mean_precision': float(np.mean(all_metrics['precision'])), 'mean_recall': float(np.mean(all_metrics['recall']))
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    return results


# ==========================================
# 核心预测函数 (自带 AMP 推理提速)
# ==========================================
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
            if y2 - y1 < window_size: y1, y2 = max(0, h - window_size), h
            if x2 - x1 < window_size: x1, x2 = max(0, w - window_size), w

            img_crop = image_tensor[:, :, y1:y2, x1:x2]
            with torch.no_grad():
              #  with torch.cuda.amp.autocast():  # 开启混合精度
                    pred_crop = torch.sigmoid(model(img_crop))

            full_probs[:, :, y1:y2, x1:x2] += pred_crop.float() * weight_map
            full_weights[:, :, y1:y2, x1:x2] += weight_map

    return full_probs / (full_weights + 1e-7)


def evaluate_model(model_path, config):
    device = config.device
    logger.info(f"开始测试模型: {model_path}")

    # ==========================================
    # 实例化 TransUNet 模型
    # ==========================================
    vit_config = CONFIGS['R50-ViT-B_16']
    vit_config.n_classes = config.num_classes
    vit_config.patches.grid = (config.crop_size // 16, config.crop_size // 16)
    model = VisionTransformer(vit_config, img_size=config.crop_size, num_classes=config.num_classes).to(device)

    # 加载你的训练权重
    checkpoint = torch.load(model_path, map_location=device)
    # 如果保存的是整个 state_dict，直接 load；如果是字典包含 'model_state' 则取内部值
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    logger.info("成功加载 TransUNet 模型权重")

    # 准备测试数据
    image_paths, mask_paths_list = get_file_list(config.test_image_dir, config.test_mask_dirs, config.mask_suffixes)

    if len(image_paths) == 0:
        logger.error("没有找到测试数据")
        return

    # 若需要随机抽取部分数据测试，可在此解开注释
    # target_count = 20
    # if len(image_paths) > target_count:
    #     indices = sorted(random.sample(range(len(image_paths)), target_count))
    #     image_paths = [image_paths[i] for i in indices]
    #     mask_paths_list = [[class_masks[i] for i in indices] for class_masks in mask_paths_list]

    test_dataset = TestDataset(image_paths, mask_paths_list, transform=get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    class_names = ['Microaneurysms', 'Haemorrhages', 'Hard_Exudates', 'Soft_Exudates']
    vis_dir = os.path.join(config.output_dir, "predictions")
    os.makedirs(vis_dir, exist_ok=True)

    all_predictions = []
    all_metrics = {k: np.zeros(config.num_classes) for k in ['iou', 'dice', 'precision', 'recall']}

    with torch.no_grad():
        for batch_idx, (images, masks, image_paths) in enumerate(tqdm(test_loader, desc="Testing")):
            images, masks = images.to(device), masks.to(device)

            # 核心推理 (直接输出 4 个通道)
            pred_masks = predict_sliding_window(
                model, images, window_size=config.crop_size, stride=config.stride, num_classes=config.num_classes
            )

            for i in range(images.shape[0]):
                # 反归一化以用于可视化
                image_np = images[i].cpu().numpy().transpose(1, 2, 0)
                mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                image_np = np.clip(image_np * std + mean, 0, 1)

                true_masks_np = masks[i].cpu().numpy()
                pred_masks_np = pred_masks[i].cpu().numpy()

                base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
                save_path = os.path.join(vis_dir, f"{base_name}_segmentation.png")

                # 生成精美对比图
                create_comparison_visualization(
                    image_np, true_masks_np, pred_masks_np, save_path, class_names, config.thresholds
                )

                # 计算指标
                sample_metrics = {'iou': [], 'dice': [], 'precision': [], 'recall': []}
                for class_idx in range(config.num_classes):
                    thresh = config.thresholds[class_idx]
                    true_binary = (true_masks_np[class_idx] > 0.5).astype(bool)
                    pred_binary = (pred_masks_np[class_idx] > thresh).astype(bool)

                    iou, dice = calculate_iou(true_binary, pred_binary), calculate_dice(true_binary, pred_binary)
                    precision, recall = calculate_precision(true_binary, pred_binary), calculate_recall(true_binary,
                                                                                                        pred_binary)

                    sample_metrics['iou'].append(iou)
                    sample_metrics['dice'].append(dice)
                    sample_metrics['precision'].append(precision)
                    sample_metrics['recall'].append(recall)

                    all_metrics['iou'][class_idx] += iou
                    all_metrics['dice'][class_idx] += dice
                    all_metrics['precision'][class_idx] += precision
                    all_metrics['recall'][class_idx] += recall

                all_predictions.append({'image_name': base_name, 'metrics': sample_metrics})

    # 计算平均指标并保存
    num_samples = len(all_predictions)
    if num_samples > 0:
        for key in all_metrics:
            all_metrics[key] /= num_samples

    json_output_path = os.path.join(config.output_dir, "test_results.json")
    save_results_to_json(all_metrics, class_names, json_output_path)

    logger.info(f"结果保存在: {vis_dir}")
    logger.info(
        f"Dice - MA: {all_metrics['dice'][0]:.4f}, HE: {all_metrics['dice'][1]:.4f}, EX: {all_metrics['dice'][2]:.4f}, SE: {all_metrics['dice'][3]:.4f}")

    return all_predictions, all_metrics


if __name__ == "__main__":
    config = TestConfig()

    # === 请在这里填入你训练好的模型权重文件路径 ===
    model_path = "E:/DR_Classification/transunet/logs/vit_final_20260324_212341.pth"

    evaluate_model(model_path, config)