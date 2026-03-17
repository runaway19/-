import os
import torch
import cv2
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


def ben_graham(image_bgr, size=512):
    img  = cv2.resize(image_bgr, (size,size))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=size/30)
    enh  = cv2.addWeighted(img, 4, blur, -4, 128)
    mask = np.zeros_like(enh)
    cv2.circle(mask, (size//2,size//2), int(size*0.45), (1,1,1), -1)
    return (enh*mask + 128*(1-mask)).astype(np.uint8)

def apply_clahe(img_rgb):
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(3.0,(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

def make_6ch(image_bgr, size=512):
    orig = cv2.cvtColor(cv2.resize(image_bgr,(size,size)), cv2.COLOR_BGR2RGB)
    enh  = apply_clahe(ben_graham(image_bgr, size))
    return np.concatenate([orig, enh], axis=2).astype(np.uint8)


class FundusClassifier(nn.Module):
    def __init__(self, num_classes=5, state_dict=None):
        super().__init__()
        base     = models.efficientnet_b0(weights=None)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            6, old_conv.out_channels,
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


def save_confusion_matrix(y_true, y_pred, names, out_dir):
    cm  = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    for ax,data,fmt,title in zip(axes,[cm,cmn],['d','.2f'],['Count','Normalized']):
        sns.heatmap(data,annot=True,fmt=fmt,cmap='Blues',
                    xticklabels=names,yticklabels=names,ax=ax)
        ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    p = os.path.join(out_dir,'confusion_matrix.png')
    plt.savefig(p,dpi=150); plt.close(); print(f"[✓] {p}")


def tta_predict(model, img6ch, val_tf, tta_tf, device, n_tta=5):
    ch1, ch2   = img6ch[:,:,:3], img6ch[:,:,3:]
    probs_list = []
    def infer(tf):
        aug = tf(image=ch1, image2=ch2)
        t   = torch.cat([aug['image'],aug['image2']],dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.softmax(model(t), dim=1)[0]
    probs_list.append(infer(val_tf))
    for _ in range(n_tta-1):
        probs_list.append(infer(tta_tf))
    avg        = torch.stack(probs_list).mean(0)
    conf, pred = torch.max(avg, dim=0)
    return pred.item(), conf.item(), avg


def batch_predict_with_gt(test_dir, gt_csv, model_path, out_dir,
                           device, image_size=512):
    os.makedirs(out_dir, exist_ok=True)
    cnames = ['正常(0)','轻度(1)','中度(2)','重度(3)','增殖期(4)']
    cshort = ['Normal','Mild','Moderate','Severe','Proliferative']
    extra  = {'additional_targets': {'image2': 'image'}}

    gt_df   = pd.read_csv(gt_csv)
    gt_dict = dict(zip(gt_df.iloc[:,0].astype(str), gt_df.iloc[:,1].astype(int)))

    sd    = torch.load(model_path, map_location=device)
    model = FundusClassifier(5, sd).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[*] 模型加载成功 | {device}")

    val_tf = A.Compose([
        A.Resize(image_size,image_size),
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ], **extra)
    tta_tf = A.Compose([
        A.Resize(image_size,image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ], **extra)

    imgs = sorted([f for f in os.listdir(test_dir)
                   if f.lower().endswith(('.jpg','.png','.jpeg','.tif'))])
    print(f"[*] 测试:{len(imgs)}张 | TTA:5次\n")

    results, preds, targets, skip = [], [], [], 0
    for name in tqdm(imgs, desc="推理"):
        fid = os.path.splitext(name)[0]
        if fid not in gt_dict: skip+=1; continue
        img = cv2.imread(os.path.join(test_dir, name))
        if img is None: skip+=1; continue
        img6          = make_6ch(img, image_size)
        pi,cf,probs   = tta_predict(model, img6, val_tf, tta_tf, device, n_tta=5)
        ri            = gt_dict[fid]
        ok            = pi == ri
        preds.append(pi); targets.append(ri)
        results.append({
            "文件名": name, "真实等级": ri, "预测等级": pi,
            "真实类别": cnames[ri], "预测类别": cnames[pi],
            "正确": "√" if ok else "✗",
            "置信度": f"{cf*100:.1f}%",
            "概率分布": " | ".join([f"C{i}:{probs[i]*100:.1f}%" for i in range(5)])
        })

    tot  = len(results)
    corr = sum(1 for r in results if r["正确"]=="√")
    qwk  = cohen_kappa_score(targets, preds, weights='quadratic')
    pc   = [0]*5; pt = [0]*5
    for t,p in zip(targets,preds):
        pt[t]+=1
        if t==p: pc[t]+=1

    pd.DataFrame(results).to_csv(
        os.path.join(out_dir,'results.csv'), index=False, encoding='utf-8-sig')
    err = pd.DataFrame(results)
    err[err["正确"]=="✗"].to_csv(
        os.path.join(out_dir,'errors.csv'), index=False, encoding='utf-8-sig')
    save_confusion_matrix(targets, preds, cshort, out_dir)

    print("\n"+"="*60)
    print(f"  总数:{tot} 跳过:{skip} | 正确:{corr} | "
          f"Acc:{corr/tot*100:.2f}% | QWK:{qwk:.4f}")
    print("-"*60)
    for i,n in enumerate(cnames):
        if pt[i]>0:
            print(f"  {n:<14}: {pc[i]:>3}/{pt[i]:<3} = {pc[i]/pt[i]*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    TEST = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    GT   = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    MDL  = "C:/Users/Administrator/Desktop/PythonProject/cls_output/best_cls_model_qwk.pth"
    OUT  = "C:/Users/Administrator/Desktop/PythonProject/cls_output"
    DEV  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_predict_with_gt(TEST, GT, MDL, OUT, DEV, image_size=512)