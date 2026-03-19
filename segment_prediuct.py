import os
import cv2
import torch
import numpy as np
import logging
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================= 配置区域 =================
class InferenceConfig:
    # 1. 输入图片文件夹路径 (存放待预测的原始图片)
    input_image_dir = "E:/DR_Classification/IDRiD_dataset/B._Disease_Grading/1._Original_Images/original_data"

    # 2. 结果输出文件夹路径 (程序会自动创建)
    output_dir = "E:/DR_Classification/IDRiD_dataset/B._Disease_Grading/1._Original_Images/reslut"

    # 3. 训练好的模型权重路径 (.pth文件)
    model_path = "E:/DR_Classification/logs/final_model_20260102_140556.pth"

    # 4. 阈值设置 (对应 [MA, HE, EX, SE])
    # 可以根据实际效果微调，建议参考 test.py 中的设置
    thresholds = [0.4, 0.5, 0.5, 0.4]

    # 5. 后缀名 (用于生成文件名)
    class_suffixes = ["_MA", "_HE", "_EX", "_SE"]

    # 6. 推理参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 640  # 滑动窗口大小 (建议与训练时的 crop_size 一致或接近)
    stride = 480  # 滑动步长 (建议为 window_size 的 3/4)


# ===========================================

def preprocess_image(image_path):
    """
    读取并预处理图像，保持与训练时一致的 CLAHE 处理
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"无法读取图像: {image_path}")
        return None, None

    # 2. CLAHE 预处理 (与 segment_main.py 保持一致)
    # 将 BGR 转为 Lab
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 仅对 L (亮度) 通道应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 合并并转回 RGB
    merged = cv2.merge((l, a, b))
    image_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    # 3. 归一化和转 Tensor (使用 albumentations)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    augmented = transform(image=image_rgb)
    image_tensor = augmented['image'].unsqueeze(0)  # 增加 Batch 维度: (1, C, H, W)

    # 返回 tensor 和原始尺寸 (用于可能的后处理，这里暂不需要)
    return image_tensor, image.shape[:2]


def predict_sliding_window(model, image_tensor, window_size=640, stride=480, num_classes=4, device='cpu'):
    """
    滑动窗口推理函数 (直接复用训练代码中的逻辑)
    """
    model.eval()
    b, c, h, w = image_tensor.shape

    # --- 1. 生成高斯权重图 ---
    def get_gaussian(window_size, sigma_scale=1.0 / 8):
        tmp = torch.arange(window_size, device=device) - window_size // 2
        sigma = window_size * sigma_scale
        gaussian1d = torch.exp(-tmp ** 2 / (2 * sigma ** 2))
        gaussian2d = torch.outer(gaussian1d, gaussian1d)
        return gaussian2d / gaussian2d.max()

    weight_map = get_gaussian(window_size).unsqueeze(0).unsqueeze(0)

    # --- 2. 初始化累加器 ---
    full_probs = torch.zeros((b, num_classes, h, w), device=device)
    full_weights = torch.zeros((b, num_classes, h, w), device=device)

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

            img_crop = image_tensor[:, :, y1:y2, x1:x2]

            with torch.no_grad():
                pred_crop = model(img_crop)
                pred_crop = torch.sigmoid(pred_crop)

            full_probs[:, :, y1:y2, x1:x2] += pred_crop * weight_map
            full_weights[:, :, y1:y2, x1:x2] += weight_map

    # --- 4. 归一化 ---
    full_probs /= (full_weights + 1e-7)
    return full_probs


def load_models(config):
    """
    加载双模型结构 (MA + Rest)
    """
    logger.info(f"正在加载模型权重: {config.model_path}")

    # 定义模型结构 (需与 segment_test.py 中保持一致)
    model_ma = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,  # 推理时不需要下载预训练权重，直接加载 checkpoint
        in_channels=3,
        classes=1,
    ).to(config.device)

    model_rest = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=3,  # HE, EX, SE
    ).to(config.device)

    # 加载权重文件
    try:
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model_ma.load_state_dict(checkpoint['model_ma_state'])
        model_rest.load_state_dict(checkpoint['model_rest_state'])
        logger.info("模型权重加载成功！")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e

    model_ma.eval()
    model_rest.eval()

    return model_ma, model_rest


def main():
    cfg = InferenceConfig()

    # 1. 准备输出目录
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
        logger.info(f"创建输出目录: {cfg.output_dir}")

    # 2. 加载模型
    model_ma, model_rest = load_models(cfg)

    # 3. 获取所有图片文件
    supported_ext = ('.jpg', '.png', '.tif', '.tiff', '.bmp')
    image_files = [f for f in os.listdir(cfg.input_image_dir) if f.lower().endswith(supported_ext)]

    if not image_files:
        logger.warning(f"在 {cfg.input_image_dir} 中未找到图片文件！")
        return

    logger.info(f"找到 {len(image_files)} 张待处理图片，开始推理...")

    # 4. 循环处理
    for img_name in tqdm(image_files, desc="Inference"):
        img_path = os.path.join(cfg.input_image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        # 4.1 预处理
        img_tensor, original_size = preprocess_image(img_path)
        if img_tensor is None:
            continue

        img_tensor = img_tensor.to(cfg.device)

        # 4.2 推理
        with torch.no_grad():
            # 预测 MA (通道 0)
            pred_ma = predict_sliding_window(
                model_ma, img_tensor,
                window_size=cfg.window_size, stride=cfg.stride,
                num_classes=1, device=cfg.device
            )

            # 预测 Rest (通道 1, 2, 3)
            pred_rest = predict_sliding_window(
                model_rest, img_tensor,
                window_size=cfg.window_size, stride=cfg.stride,
                num_classes=3, device=cfg.device
            )

            # 拼接结果 -> (1, 4, H, W)
            full_preds = torch.cat([pred_ma, pred_rest], dim=1)

        # 4.3 后处理与保存
        # 转为 numpy: (4, H, W)
        probs_np = full_preds[0].cpu().numpy()

        for i, suffix in enumerate(cfg.class_suffixes):
            # 获取当前类别的概率图
            prob_map = probs_np[i]

            # 二值化 (0 或 255)
            binary_mask = (prob_map > cfg.thresholds[i]).astype(np.uint8) * 255

            # 生成文件名: 图像名 + 病灶名.png
            save_name = f"{base_name}{suffix}.png"
            save_path = os.path.join(cfg.output_dir, save_name)

            # 保存
            cv2.imwrite(save_path, binary_mask)

    logger.info(f"推理完成！结果已保存在: {cfg.output_dir}")


if __name__ == "__main__":
    main()