import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp


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
    print(f"[*] 分割模型加载成功")
    return model_ma, model_rest


def seg_sliding_window(model, t, ws, stride, nc, device):
    b, c, h, w = t.shape
    tmp  = torch.arange(ws, device=device) - ws // 2
    g1d  = torch.exp(-tmp**2 / (2*(ws/8)**2))
    wmap = (torch.outer(g1d, g1d) / torch.outer(g1d, g1d).max()).unsqueeze(0).unsqueeze(0)
    fp   = torch.zeros((b, nc, h, w), device=device)
    fw   = torch.zeros((b, nc, h, w), device=device)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2 = min(y + ws, h); x2 = min(x + ws, w)
            if y2 - y1 < ws: y1 = max(0, h - ws); y2 = h
            if x2 - x1 < ws: x1 = max(0, w - ws); x2 = w
            with torch.no_grad():
                pred = torch.sigmoid(model(t[:, :, y1:y2, x1:x2]))
            fp[:, :, y1:y2, x1:x2] += pred * wmap
            fw[:, :, y1:y2, x1:x2] += wmap
    return fp / (fw + 1e-7)


def get_seg_masks(img_rgb, model_ma, model_rest, device,
                   ws=640, stride=480, thresholds=None):
    if thresholds is None:
        thresholds = [0.4, 0.5, 0.5, 0.4]
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    img_c = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    tf = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    t  = tf(image=img_c)['image'].unsqueeze(0).to(device)
    pm = seg_sliding_window(model_ma,   t, ws, stride, 1, device)
    pr = seg_sliding_window(model_rest, t, ws, stride, 3, device)
    fp = torch.cat([pm, pr], dim=1)[0]
    masks = [(fp[i].cpu().numpy() > thresholds[i]).astype(np.float32)
             for i in range(4)]
    return np.stack(masks, axis=2)   # (H, W, 4)


# ================= 预处理 =================
def ben_graham(image_bgr, size=512):
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
    l       = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def make_10ch_tensor(image_bgr, model_ma, model_rest, device,
                      size=512, seg_thresholds=None):
    """
    构建10通道tensor (10, H, W)：
    0-2: 原图RGB归一化
    3-5: Ben Graham增强RGB归一化
    6-9: 分割mask (0或1, float)
    """
    orig_rgb = cv2.cvtColor(cv2.resize(image_bgr, (size, size)), cv2.COLOR_BGR2RGB)
    enh_rgb  = apply_clahe(ben_graham(image_bgr, size))

    # 分割mask（在全尺寸图上推理再resize）
    full_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks    = get_seg_masks(full_rgb, model_ma, model_rest, device,
                              thresholds=seg_thresholds)             # (H,W,4)
    masks_r  = cv2.resize(masks, (size, size),
                           interpolation=cv2.INTER_NEAREST)          # (size,size,4)

    norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t1   = torch.from_numpy(
        np.transpose(norm(image=orig_rgb)['image'], (2, 0, 1))).float()  # (3,H,W)
    t2   = torch.from_numpy(
        np.transpose(norm(image=enh_rgb)['image'],  (2, 0, 1))).float()  # (3,H,W)
    t3   = torch.from_numpy(
        np.transpose(masks_r, (2, 0, 1))).float()                        # (4,H,W)

    return torch.cat([t1, t2, t3], dim=0)   # (10,H,W)


# ================= 分类模型 =================
class FundusClassifier(nn.Module):
    def __init__(self, num_classes=5, state_dict=None):
        super().__init__()
        base     = models.efficientnet_b0(weights=None)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            10, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding, bias=False)
        base.features[0][0] = new_conv
        num_features  = base.classifier[1].in_features
        self.features = base.features
        self.avgpool  = base.avgpool
        hidden = 256
        if state_dict and "classifier.2.weight" in state_dict:
            hidden = state_dict["classifier.2.weight"].shape[0]
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.4),
            nn.Linear(num_features, hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        return self.classifier(torch.flatten(feat, 1))


# ================= 混淆矩阵 =================
def save_confusion_matrix(y_true, y_pred, names, out_dir):
    cm  = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
        axes, [cm, cmn], ['d', '.2f'], ['Count', 'Normalized']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    p = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(p, dpi=150); plt.close(); print(f"[✓] {p}")


# ================= TTA推理（修复版）=================
def tta_predict(model, base_tensor, device, n_tta=5):
    """
    对10通道tensor做TTA：
    - 对所有通道做相同的几何变换（翻转/旋转）
    - 不再做归一化（输入已经归一化好了）
    - mask通道跟着几何变换同步变化
    """
    probs_list = []

    def infer(t):
        with torch.no_grad():
            return torch.softmax(model(t.unsqueeze(0).to(device)), dim=1)[0]

    # 第1次：原始输入
    probs_list.append(infer(base_tensor))

    # 后续TTA：对10通道numpy做几何变换
    arr = base_tensor.numpy()   # (10, H, W)

    tta_ops = [
        lambda x: np.flip(x, axis=2),                       # 水平翻转
        lambda x: np.flip(x, axis=1),                       # 垂直翻转
        lambda x: np.rot90(x, k=1, axes=(1, 2)),            # 旋转90°
        lambda x: np.flip(np.flip(x, axis=2), axis=1),      # 水平+垂直
    ]

    for i, op in enumerate(tta_ops[:n_tta - 1]):
        arr_aug = np.ascontiguousarray(op(arr))
        t_aug   = torch.from_numpy(arr_aug).float()
        probs_list.append(infer(t_aug))

    avg        = torch.stack(probs_list).mean(0)
    conf, pred = torch.max(avg, dim=0)
    return pred.item(), conf.item(), avg


# ================= 主推理 =================
def batch_predict_with_gt(test_dir, gt_csv, cls_model_path, seg_model_path,
                           out_dir, device, image_size=512,
                           seg_thresholds=None):
    if seg_thresholds is None:
        seg_thresholds = [0.4, 0.5, 0.5, 0.4]

    os.makedirs(out_dir, exist_ok=True)
    cnames = ['正常(0)', '轻度(1)', '中度(2)', '重度(3)', '增殖期(4)']
    cshort = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    gt_df   = pd.read_csv(gt_csv)
    gt_dict = dict(zip(gt_df.iloc[:,0].astype(str), gt_df.iloc[:,1].astype(int)))

    # 加载分割模型
    print("[*] 加载分割模型...")
    model_ma, model_rest = load_seg_model(seg_model_path, device)

    # 加载分类模型
    sd    = torch.load(cls_model_path, map_location=device)
    model = FundusClassifier(5, sd).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[*] 分类模型加载成功 | {device}")

    imgs = sorted([f for f in os.listdir(test_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))])
    print(f"[*] 测试:{len(imgs)}张 | TTA:5次\n")

    results, preds, targets, skip = [], [], [], 0

    for name in tqdm(imgs, desc="推理"):
        fid = os.path.splitext(name)[0]
        if fid not in gt_dict: skip += 1; continue
        img = cv2.imread(os.path.join(test_dir, name))
        if img is None: skip += 1; continue

        # 构建10通道tensor（CPU上）
        t10 = make_10ch_tensor(img, model_ma, model_rest, device,
                                size=image_size,
                                seg_thresholds=seg_thresholds)

        pi, cf, probs = tta_predict(model, t10, device, n_tta=5)
        ri            = gt_dict[fid]
        ok            = pi == ri
        preds.append(pi); targets.append(ri)
        results.append({
            "文件名":   name,
            "真实等级": ri,
            "预测等级": pi,
            "真实类别": cnames[ri],
            "预测类别": cnames[pi],
            "正确":     "√" if ok else "✗",
            "置信度":   f"{cf*100:.1f}%",
            "概率分布": " | ".join([f"C{i}:{probs[i]*100:.1f}%" for i in range(5)])
        })

    tot  = len(results)
    corr = sum(1 for r in results if r["正确"] == "√")
    qwk  = cohen_kappa_score(targets, preds, weights='quadratic')
    pc   = [0]*5; pt = [0]*5
    for t, p in zip(targets, preds):
        pt[t] += 1
        if t == p: pc[t] += 1

    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, 'results.csv'), index=False, encoding='utf-8-sig')
    err = pd.DataFrame(results)
    err[err["正确"] == "✗"].to_csv(
        os.path.join(out_dir, 'errors.csv'), index=False, encoding='utf-8-sig')
    save_confusion_matrix(targets, preds, cshort, out_dir)

    print("\n" + "="*60)
    print(f"  总数:{tot} 跳过:{skip} | 正确:{corr} | "
          f"Acc:{corr/tot*100:.2f}% | QWK:{qwk:.4f}")
    print("-"*60)
    for i, n in enumerate(cnames):
        if pt[i] > 0:
            print(f"  {n:<14}: {pc[i]:>3}/{pt[i]:<3} = {pc[i]/pt[i]*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    TEST    = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    GT      = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    CLS_MDL = "C:/Users/Administrator/Desktop/PythonProject/cls_output/best_cls_model_qwk.pth"
    SEG_MDL = "C:/Users/Administrator/Desktop/PythonProject/final_model_20260102_140556.pth"
    OUT     = "C:/Users/Administrator/Desktop/PythonProject/cls_output"
    DEV     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_predict_with_gt(TEST, GT, CLS_MDL, SEG_MDL, OUT, DEV,
                           image_size=512,
                           seg_thresholds=[0.4, 0.5, 0.5, 0.4])