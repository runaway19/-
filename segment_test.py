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
import segmentation_models_pytorch as smp

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

class TestConfig:
    # 测试数据路径 - 修改为您的测试数据路径
    test_image_dir = "E:/DR_Classification/IDRiD_dataset/A._Segmentation/1._Original_Images/b._Testing_Set"
    test_mask_dirs = [
        "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/b._Testing_Set/1._Microaneurysms",
        "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/b._Testing_Set/2._Haemorrhages",
        "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/b._Testing_Set/3._Hard_Exudates",
        "E:/DR_Classification/IDRiD_dataset/A._Segmentation/2._All_Segmentation_Groundtruths/b._Testing_Set/4._Soft_Exudates"
    ]

    # 每个mask目录对应的后缀
    mask_suffixes = ["_MA", "_HE", "_EX", "_SE"]

    # 模型参数
    batch_size = 2
    num_classes = 4
    input_channels = 3
    thresholds = [0.4, 0.5, 0.5, 0.4]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 结果保存路径
    output_dir = "E:/DR_Classification/outputs/sement_reslut"
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        original_height, original_width = image.shape[:2]

        # 读取五个mask，处理缺失的情况
        masks = []
        for i, mask_paths in enumerate(self.mask_paths_list):
            mask_path = mask_paths[idx]

            if mask_path is None:
                mask = np.zeros((original_height, original_width), dtype=np.float32)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    try:
                        mask = plt.imread(mask_path)
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                        mask = (mask * 255).astype(np.uint8)
                    except:
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
            mask_tensors = [augmented[f'mask{i}'] for i in range(len(masks))]
            mask_tensor = torch.stack(mask_tensors, dim=0)
        else:
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
            multi_mask = np.stack(masks, axis=0)
            mask_tensor = torch.from_numpy(multi_mask).float()

        return image_tensor, mask_tensor, image_path


def get_test_transforms():
    """测试时的数据变换"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={f'mask{i}': 'mask' for i in range(4)})


def get_file_list(image_dir, mask_dirs, mask_suffixes):
    """获取匹配的图像和mask文件列表 - 复用训练代码中的函数"""
    # 获取图像文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
    logger.info(f"找到 {len(image_files)} 个测试图像文件")

    # 为每个mask目录获取文件列表
    mask_files_list = []
    for mask_dir in mask_dirs:
        if not os.path.exists(mask_dir):
            logger.warning(f"Mask目录不存在: {mask_dir}")
            mask_files = []
        else:
            mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]
        logger.info(f"在 {os.path.basename(mask_dir)} 中找到 {len(mask_files)} 个mask文件")
        mask_files_list.append(mask_files)

    # 构建图像基础名称到文件路径的映射
    image_base_to_path = {}
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        image_base_to_path[base_name] = os.path.join(image_dir, img_file)

    # 为每个mask目录构建基础名称到文件路径的映射
    mask_base_to_path_list = []
    for i, (mask_files, suffix) in enumerate(zip(mask_files_list, mask_suffixes)):
        mask_base_to_path = {}
        for mask_file in mask_files:
            base_name = os.path.splitext(mask_file)[0]
            # 移除特定的后缀来匹配图像文件名
            if base_name.endswith(suffix):
                base_name_without_suffix = base_name[:-len(suffix)]
                mask_base_to_path[base_name_without_suffix] = os.path.join(mask_dirs[i], mask_file)
            else:
                # 如果没有预期的后缀，直接使用
                mask_base_to_path[base_name] = os.path.join(mask_dirs[i], mask_file)
        mask_base_to_path_list.append(mask_base_to_path)

    # 找到所有图像基础名称的并集
    all_image_basenames = set(image_base_to_path.keys())

    # 构建路径列表 - 允许mask路径为None
    image_paths = []
    mask_paths_list = [[] for _ in range(len(mask_dirs))]

    # 统计每个类别的可用mask数量
    class_counts = [0] * len(mask_dirs)

    for base_name in sorted(all_image_basenames):
        # 检查这个图像是否有至少一个mask
        has_at_least_one_mask = False

        for i, mask_base_to_path in enumerate(mask_base_to_path_list):
            if base_name in mask_base_to_path:
                mask_path = mask_base_to_path[base_name]
                mask_paths_list[i].append(mask_path)
                class_counts[i] += 1
                has_at_least_one_mask = True
            else:
                mask_paths_list[i].append(None)

        # 只有当图像至少有一个mask时，才将其加入测试集
        if has_at_least_one_mask:
            image_paths.append(image_base_to_path[base_name])
        else:
            # 如果图像没有任何mask，从所有mask路径列表中移除对应的None
            for i in range(len(mask_dirs)):
                mask_paths_list[i].pop()

    # 打印每个类别的mask数量统计
    for i, count in enumerate(class_counts):
        logger.info(f"类别 {i} ({mask_suffixes[i]}): {count} 个mask")

    logger.info(f"最终使用 {len(image_paths)} 个测试样本")

    return image_paths, mask_paths_list


def create_comparison_visualization(image, true_masks, pred_masks, save_path, class_names, thresholds):
    """
    [修复版] 优化可视化：使用 GridSpec 解决 tight_layout 警告问题
    包含全类别合成图 + 详细的误差分析(TP/FP/FN)
    """
    # 1. 准备底图
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_vis = image.astype(np.uint8).copy()

    # 2. 定义画布和 GridSpec
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 12)

    # 1.1 原图
    ax1 = fig.add_subplot(gs[0, 0:4])
    ax1.imshow(image_vis)
    ax1.set_title("Original Image", fontsize=12, weight='bold')
    ax1.axis('off')

    # 定义类别颜色 (R, G, B)
    class_colors = [
        (255, 0, 0),  # MA: Red
        (0, 255, 0),  # HE: Green
        (0, 0, 255),  # EX: Blue
        (255, 255, 0)  # SE: Yellow
    ]

    # 1.2 Ground Truth 合成图
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

    # 1.3 Prediction 合成图
    pred_overlay = image_vis.copy()
    pred_legend = []
    for i, color in enumerate(class_colors):
        mask = (pred_masks[i] > thresholds[i]).astype(bool)

        if mask.sum() > 0:
            colored_mask = np.zeros_like(image_vis)
            colored_mask[mask] = color
            mask_indices = np.any(colored_mask > 0, axis=-1)
            pred_overlay[mask_indices] = cv2.addWeighted(image_vis[mask_indices], 0.6, colored_mask[mask_indices], 0.4,
                                                         0)
            pred_legend.append(f"{class_names[i]}")

    ax3 = fig.add_subplot(gs[0, 8:12])
    ax3.imshow(pred_overlay)
    ax3.set_title(f"Prediction (Merged)", fontsize=12, weight='bold')
    ax3.axis('off')

    # === 第二部分：单类别误差分析 (Row 2) ===
    for i in range(4):
        col_start = i * 3
        col_end = (i + 1) * 3
        ax = fig.add_subplot(gs[1, col_start:col_end])

        # 准备 Masks
        t_mask = (true_masks[i] > 0.5).astype(bool)

        p_mask = (pred_masks[i] > thresholds[i]).astype(bool)

        # 计算 TP, FP, FN
        tp = t_mask & p_mask
        fp = (~t_mask) & p_mask
        fn = t_mask & (~p_mask)

        # 绘制误差图
        error_overlay = image_vis.copy()
        error_overlay[tp] = [0, 255, 0]  # Green: Correct
        error_overlay[fp] = [255, 0, 0]  # Red: False Positive
        error_overlay[fn] = [0, 0, 255]  # Blue: False Negative (Missed)

        intersection = tp.sum()
        union = t_mask.sum() + p_mask.sum()
        dice = (2.0 * intersection) / union if union > 0 else 0.0

        ax.imshow(error_overlay)
        # 显示当前使用的具体阈值
        ax.set_title(f"{class_names[i]} (T={thresholds[i]})\nDice: {dice:.3f}", fontsize=10, weight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_iou(true_mask, pred_mask):
    """计算IoU指标"""
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0


def calculate_dice(true_mask, pred_mask):
    """计算Dice系数"""
    intersection = np.logical_and(true_mask, pred_mask).sum()
    total = true_mask.sum() + pred_mask.sum()
    return (2.0 * intersection) / total if total > 0 else 0.0


def calculate_precision(true_mask, pred_mask):
    """计算精确率"""
    true_positives = np.logical_and(true_mask, pred_mask).sum()
    false_positives = np.logical_and(~true_mask, pred_mask).sum()
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0


def calculate_recall(true_mask, pred_mask):
    """计算召回率"""
    true_positives = np.logical_and(true_mask, pred_mask).sum()
    false_negatives = np.logical_and(true_mask, ~pred_mask).sum()
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0


def save_results_to_json(all_metrics, class_names, output_path):
    """将结果保存到JSON文件"""
    results = {
        'overall_metrics': {},
        'class_metrics': {}
    }

    # 按类别保存指标
    for i, class_name in enumerate(class_names):
        results['class_metrics'][class_name] = {
            'iou': float(all_metrics['iou'][i]),
            'dice': float(all_metrics['dice'][i]),
            'precision': float(all_metrics['precision'][i]),
            'recall': float(all_metrics['recall'][i])
        }

    # 计算总体平均值
    results['overall_metrics'] = {
        'mean_iou': float(np.mean(all_metrics['iou'])),
        'mean_dice': float(np.mean(all_metrics['dice'])),
        'mean_precision': float(np.mean(all_metrics['precision'])),
        'mean_recall': float(np.mean(all_metrics['recall']))
    }

    # 保存到文件
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"结果已保存到: {output_path}")
    return results


def predict_sliding_window(model, image_tensor, window_size=512, stride=384, num_classes=4):
    """
    使用高斯加权滑动窗口进行全图推理 (消除网格伪影)
    """
    model.eval()
    b, c, h, w = image_tensor.shape
    device = image_tensor.device

    # --- 1. 生成高斯权重图 ---
    # 这种权重图中间高(1.0)，四周低(接近0)，形状像个土包
    def get_gaussian(window_size, sigma_scale=1.0 / 8):
        tmp = torch.arange(window_size, device=device) - window_size // 2
        sigma = window_size * sigma_scale
        # 生成 1D 高斯
        gaussian1d = torch.exp(-tmp ** 2 / (2 * sigma ** 2))
        # 生成 2D 高斯 (外积)
        gaussian2d = torch.outer(gaussian1d, gaussian1d)
        # 归一化到 0-1
        return gaussian2d / gaussian2d.max()

    # 初始化权重窗口 (1, 1, H, W)
    weight_map = get_gaussian(window_size).unsqueeze(0).unsqueeze(0)

    # --- 2. 初始化累加器 ---
    full_probs = torch.zeros((b, num_classes, h, w), device=device)
    full_weights = torch.zeros((b, num_classes, h, w), device=device)  # 记录权重的累加值

    # --- 3. 滑动窗口循环 ---
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            x1 = x
            y2 = min(y + window_size, h)
            x2 = min(x + window_size, w)

            # 边界修正
            if y2 - y1 < window_size:
                y1 = max(0, h - window_size)
                y2 = h
            if x2 - x1 < window_size:
                x1 = max(0, w - window_size)
                x2 = w

            # 切图
            img_crop = image_tensor[:, :, y1:y2, x1:x2]

            with torch.no_grad():
                pred_crop = model(img_crop)
                pred_crop = torch.sigmoid(pred_crop)

            # --- 核心修改：累加时乘上高斯权重 ---
            # 这里的 weight_map 对应当前 crop 的权重
            full_probs[:, :, y1:y2, x1:x2] += pred_crop * weight_map
            full_weights[:, :, y1:y2, x1:x2] += weight_map

    # --- 4. 归一化 ---
    # 总预测值 / 总权重值
    # 加上 epsilon 防止除零
    full_probs /= (full_weights + 1e-7)

    return full_probs

def evaluate_model(model_path, config):
    """测试训练好的模型"""
    device = config.device

    logger.info(f"开始测试模型: {model_path}")
    logger.info(f"使用设备: {config.device}")

    # 模型 A: 专注 MA (UNet++)
    model_ma = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    model_rest = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",  # 或者 resnet34，efficientnet-b0参数更少训练更快
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,  # HE, EX, SE
    ).to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=config.device)

    model_ma.load_state_dict(checkpoint['model_ma_state'])
    model_rest.load_state_dict(checkpoint['model_rest_state'])

    model_ma.eval()
    model_rest.eval()
    logger.info(f"成功加载模型权重，训练轮次: {checkpoint.get('epoch', '未知')}")

    # 准备测试数据
    image_paths, mask_paths_list = get_file_list(
        config.test_image_dir,
        config.test_mask_dirs,
        config.mask_suffixes
    )

    if len(image_paths) == 0:
        logger.error("没有找到测试数据")
        return

    target_count = 40

    if len(image_paths) > target_count:
        logger.info(f"数据量 ({len(image_paths)}) 超过 {target_count}，正在随机抽取...")

        # 1. 设置随机种子以保证结果可复现 (可选)
        random.seed(42)

        # 2. 生成随机索引
        # random.sample 会从 range 中无放回地抽取 target_count 个数
        indices = sorted(random.sample(range(len(image_paths)), target_count))

        # 3. 根据索引筛选 image_paths
        image_paths = [image_paths[i] for i in indices]

        # 4. 根据索引筛选 mask_paths_list
        # 注意：mask_paths_list 是一个 list of lists (4个类别列表)，需要分别筛选
        new_mask_paths_list = []
        for class_masks in mask_paths_list:
            # class_masks 是某一类的所有 mask 路径列表
            new_mask_paths_list.append([class_masks[i] for i in indices])

        mask_paths_list = new_mask_paths_list

        logger.info(f"随机抽样完成，当前测试集大小: {len(image_paths)}")
    else:
        logger.info(f"数据量 ({len(image_paths)}) 不足 {target_count}，使用全部数据进行测试")
    # ============================================================

    logger.info(f"找到 {len(image_paths)} 个测试样本")

    # 创建测试数据集和数据加载器
    test_dataset = TestDataset(
        image_paths,
        mask_paths_list,
        transform=get_test_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # 类别名称
    class_names = ['Microaneurysms', 'Haemorrhages', 'Hard_Exudates', 'Soft_Exudates']

    # 创建可视化结果目录
    vis_dir = os.path.join(config.output_dir, "predictions")
    os.makedirs(vis_dir, exist_ok=True)

    # 存储所有预测结果和指标
    all_predictions = []
    all_metrics = {
        'iou': np.zeros(config.num_classes),
        'dice': np.zeros(config.num_classes),
        'precision': np.zeros(config.num_classes),
        'recall': np.zeros(config.num_classes)
    }

    with torch.no_grad():
        for batch_idx, (images, masks, image_paths) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(config.device)
            masks = masks.to(config.device)

            # === 修改 3: 分别推理并合并结果 ===

            # 3.1 预测 MA (1通道)
            pred_ma = predict_sliding_window(
                model_ma,
                images,
                window_size=512,
                stride=384,
                num_classes=1  # 指定为 1
            )

            # 3.2 预测其他 (3通道)
            pred_rest = predict_sliding_window(
                model_rest,
                images,
                window_size=512,
                stride=384,
                num_classes=3  # 指定为 3
            )

            # 3.3 拼接结果 (Batch, 4, H, W)
            # 顺序必须是: MA(0) + [HE(1), EX(2), SE(3)]
            pred_masks = torch.cat([pred_ma, pred_rest], dim=1)

            # === 以下代码保持不变 ===
            # 保存每个样本的结果
            for i in range(images.shape[0]):
                image_np = images[i].cpu().numpy()
                image_np = np.transpose(image_np, (1, 2, 0))
                # 反归一化
                mean = np.array([0.485, 0.456, 0.406])  # 注意：这里要用你 transform 里设置的 mean
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)

                # 获取真实mask和预测mask
                true_masks_np = masks[i].cpu().numpy()
                pred_masks_np = pred_masks[i].cpu().numpy()

                base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
                save_path = os.path.join(vis_dir, f"{base_name}_segmentation.png")

                # === 修改点 1: 传入 config.thresholds 列表 ===
                create_comparison_visualization(
                    image_np,
                    true_masks_np,
                    pred_masks_np,
                    save_path,
                    class_names,
                    thresholds=config.thresholds  # <--- 使用配置列表
                )

                # 计算每个类别的指标
                sample_metrics = {'iou': [], 'dice': [], 'precision': [], 'recall': []}
                for class_idx in range(config.num_classes):
                    # 针对 MA (class_idx == 0) 可以使用较低的阈值
                    thresh = config.thresholds[class_idx]

                    true_binary = (true_masks_np[class_idx] > 0.5).astype(bool)
                    pred_binary = (pred_masks_np[class_idx] > thresh).astype(bool)

                    iou = calculate_iou(true_binary, pred_binary)
                    dice = calculate_dice(true_binary, pred_binary)
                    precision = calculate_precision(true_binary, pred_binary)
                    recall = calculate_recall(true_binary, pred_binary)

                    sample_metrics['iou'].append(iou)
                    sample_metrics['dice'].append(dice)
                    sample_metrics['precision'].append(precision)
                    sample_metrics['recall'].append(recall)

                    # 累加到总体指标
                    all_metrics['iou'][class_idx] += iou
                    all_metrics['dice'][class_idx] += dice
                    all_metrics['precision'][class_idx] += precision
                    all_metrics['recall'][class_idx] += recall

                # 保存预测的mask为npy文件
                npy_save_path = os.path.join(vis_dir, f"{base_name}_pred_masks.npy")
                np.save(npy_save_path, pred_masks_np)

                all_predictions.append({
                    'image_name': base_name,
                    'metrics': sample_metrics
                })

    # 计算平均指标
    num_samples = len(all_predictions)
    if num_samples > 0:
        for key in all_metrics:
            all_metrics[key] /= num_samples

    # 保存结果到JSON文件
    json_output_path = os.path.join(config.output_dir, "test_results.json")
    json_results = save_results_to_json(all_metrics, class_names, json_output_path)

    logger.info(f"测试完成！结果保存在: {vis_dir}")
    logger.info(f"共处理 {len(all_predictions)} 个测试样本")

    return all_predictions, all_metrics, json_results


if __name__ == "__main__":
    config = TestConfig()

    # 指定您的模型权重文件路径
    model_path = "E:/DR_Classification/logs/final_model_20260108_225120.pth"  # 修改为您的模型文件路径

    # 运行测试
    results = evaluate_model(model_path, config)