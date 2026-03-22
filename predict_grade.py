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
    print("[*] 分割模型加载成功")
    return model_ma, model_rest

def seg_sliding_window(model, t, ws, stride, nc, device):
    b,c,h,w = t.shape
    tmp  = torch.arange(ws,device=device) - ws//2
    g1d  = torch.exp(-tmp**2/(2*(ws/8)**2))
    wmap = (torch.outer(g1d,g1d)/torch.outer(g1d,g1d).max()).unsqueeze(0).unsqueeze(0)
    fp   = torch.zeros((b,nc,h,w),device=device)
    fw   = torch.zeros((b,nc,h,w),device=device)
    for y in range(0,h,stride):
        for x in range(0,w,stride):
            y1,x1=y,x; y2=min(y+ws,h); x2=min(x+ws,w)
            if y2-y1<ws: y1=max(0,h-ws); y2=h
            if x2-x1<ws: x1=max(0,w-ws); x2=w
            with torch.no_grad():
                pred=torch.sigmoid(model(t[:,:,y1:y2,x1:x2]))
            fp[:,:,y1:y2,x1:x2]+=pred*wmap; fw[:,:,y1:y2,x1:x2]+=wmap
    return fp/(fw+1e-7)

def get_seg_masks(img_rgb, model_ma, model_rest, device,
                   ws=640, stride=480, thresholds=None):
    if thresholds is None: thresholds=[0.4,0.5,0.5,0.4]
    lab=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(2.0,(8,8)).apply(l)
    img_c=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2RGB)
    tf=A.Compose([A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),ToTensorV2()])
    t=tf(image=img_c)['image'].unsqueeze(0).to(device)
    pm=seg_sliding_window(model_ma,t,ws,stride,1,device)
    pr=seg_sliding_window(model_rest,t,ws,stride,3,device)
    fp=torch.cat([pm,pr],dim=1)[0]
    return np.stack([(fp[i].cpu().numpy()>thresholds[i]).astype(np.float32)
                     for i in range(4)],axis=2)


# ================= 预处理 =================
def ben_graham(image_bgr, size=512):
    img=cv2.resize(image_bgr,(size,size)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    blur=cv2.GaussianBlur(img,(0,0),sigmaX=size/30)
    enh=cv2.addWeighted(img,4,blur,-4,128)
    mask=np.zeros_like(enh); cv2.circle(mask,(size//2,size//2),int(size*0.45),(1,1,1),-1)
    return (enh*mask+128*(1-mask)).astype(np.uint8)

def apply_clahe(img_rgb):
    lab=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(3.0,(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2RGB)

def prepare_inputs(image_bgr, model_ma, model_rest, device,
                    size=512, seg_thresholds=None):
    orig_rgb = cv2.cvtColor(cv2.resize(image_bgr,(size,size)),cv2.COLOR_BGR2RGB)
    enh_rgb  = apply_clahe(ben_graham(image_bgr,size))
    full_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    masks    = get_seg_masks(full_rgb,model_ma,model_rest,device,
                              thresholds=seg_thresholds)
    masks_r  = cv2.resize(masks,(size,size),interpolation=cv2.INTER_NEAREST)
    norm  = A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    t1    = torch.from_numpy(np.transpose(norm(image=orig_rgb)['image'],(2,0,1))).float()
    t2    = torch.from_numpy(np.transpose(norm(image=enh_rgb)['image'],(2,0,1))).float()
    t_img = torch.cat([t1,t2],dim=0).unsqueeze(0).to(device)
    t_msk = torch.from_numpy(np.transpose(masks_r,(2,0,1))).float().unsqueeze(0).to(device)
    return t_img, t_msk


# ================= 简化分类模型V2 =================
class FundusClassifierV2(nn.Module):
    def __init__(self, num_classes=5, state_dict=None):
        super().__init__()
        base     = models.efficientnet_b0(weights=None)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(6,old_conv.out_channels,kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,padding=old_conv.padding,bias=False)
        base.features[0][0] = new_conv
        self.img_backbone = base.features
        self.img_pool     = base.avgpool
        img_feat_dim      = base.classifier[1].in_features

        self.area_embed = nn.Sequential(
            nn.Linear(4,32), nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32,64), nn.ReLU(inplace=True),
        )

        fused_dim = img_feat_dim + 64
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim), nn.Dropout(p=0.45),
            nn.Linear(fused_dim,256), nn.ReLU(inplace=True),
            nn.BatchNorm1d(256), nn.Dropout(p=0.3),
            nn.Linear(256,num_classes),
        )

    def forward(self, img6, mask4):
        img_feat  = torch.flatten(self.img_pool(self.img_backbone(img6)),1)
        area      = mask4.mean(dim=(2,3))
        area_feat = self.area_embed(area)
        return self.classifier(torch.cat([img_feat,area_feat],dim=1))


# ================= 混淆矩阵 =================
def save_confusion_matrix(y_true, y_pred, names, out_dir):
    cm  = confusion_matrix(y_true,y_pred)
    cmn = cm.astype(float)/cm.sum(axis=1,keepdims=True)
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for ax,data,fmt,title in zip(axes,[cm,cmn],['d','.2f'],['Count','Normalized']):
        sns.heatmap(data,annot=True,fmt=fmt,cmap='Blues',
                    xticklabels=names,yticklabels=names,ax=ax)
        ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    p=os.path.join(out_dir,'confusion_matrix.png')
    plt.savefig(p,dpi=150); plt.close(); print(f"[✓] {p}")


# ================= TTA推理 =================
def tta_predict(model, img6_base, mask4_base, device, n_tta=5):
    probs_list = []
    def infer(img6, mask4):
        with torch.no_grad():
            return torch.softmax(model(img6,mask4),dim=1)[0]
    probs_list.append(infer(img6_base, mask4_base))
    img_np  = img6_base[0].cpu().numpy()
    mask_np = mask4_base[0].cpu().numpy()
    tta_ops = [
        lambda x: np.flip(x,axis=2),
        lambda x: np.flip(x,axis=1),
        lambda x: np.rot90(x,k=1,axes=(1,2)),
        lambda x: np.flip(np.flip(x,axis=2),axis=1),
    ]
    for op in tta_ops[:n_tta-1]:
        img_aug  = torch.from_numpy(np.ascontiguousarray(op(img_np))).float().unsqueeze(0).to(device)
        mask_aug = torch.from_numpy(np.ascontiguousarray(op(mask_np))).float().unsqueeze(0).to(device)
        probs_list.append(infer(img_aug,mask_aug))
    avg        = torch.stack(probs_list).mean(0)
    conf, pred = torch.max(avg,dim=0)
    return pred.item(), conf.item(), avg


# ================= 主推理 =================
def batch_predict_with_gt(test_dir, gt_csv, cls_model_path, seg_model_path,
                           out_dir, device, image_size=512, seg_thresholds=None):
    if seg_thresholds is None: seg_thresholds=[0.4,0.5,0.5,0.4]
    os.makedirs(out_dir,exist_ok=True)
    cnames=['正常(0)','轻度(1)','中度(2)','重度(3)','增殖期(4)']
    cshort=['Normal','Mild','Moderate','Severe','Proliferative']

    gt_df   = pd.read_csv(gt_csv)
    gt_dict = dict(zip(gt_df.iloc[:,0].astype(str),gt_df.iloc[:,1].astype(int)))

    print("[*] 加载分割模型...")
    model_ma,model_rest = load_seg_model(seg_model_path,device)

    sd    = torch.load(cls_model_path,map_location=device)
    model = FundusClassifierV2(5,sd).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[*] 分类模型加载成功 | {device}")

    imgs = sorted([f for f in os.listdir(test_dir)
                   if f.lower().endswith(('.jpg','.png','.jpeg','.tif'))])
    print(f"[*] 测试:{len(imgs)}张 | TTA:5次\n")

    results,preds,targets,skip=[],[],[],0
    for name in tqdm(imgs,desc="推理"):
        fid=os.path.splitext(name)[0]
        if fid not in gt_dict: skip+=1; continue
        img=cv2.imread(os.path.join(test_dir,name))
        if img is None: skip+=1; continue
        img6,mask4=prepare_inputs(img,model_ma,model_rest,device,
                                   size=image_size,seg_thresholds=seg_thresholds)
        pi,cf,probs=tta_predict(model,img6,mask4,device,n_tta=5)
        ri=gt_dict[fid]; ok=pi==ri
        preds.append(pi); targets.append(ri)
        results.append({
            "文件名":name,"真实等级":ri,"预测等级":pi,
            "真实类别":cnames[ri],"预测类别":cnames[pi],
            "正确":"√" if ok else "✗",
            "置信度":f"{cf*100:.1f}%",
            "概率分布":" | ".join([f"C{i}:{probs[i]*100:.1f}%" for i in range(5)])
        })

    tot=len(results); corr=sum(1 for r in results if r["正确"]=="√")
    qwk=cohen_kappa_score(targets,preds,weights='quadratic')
    pc=[0]*5; pt=[0]*5
    for t,p in zip(targets,preds):
        pt[t]+=1
        if t==p: pc[t]+=1

    pd.DataFrame(results).to_csv(os.path.join(out_dir,'results.csv'),
                                  index=False,encoding='utf-8-sig')
    err=pd.DataFrame(results); err[err["正确"]=="✗"].to_csv(
        os.path.join(out_dir,'errors.csv'),index=False,encoding='utf-8-sig')
    save_confusion_matrix(targets,preds,cshort,out_dir)

    print("\n"+"="*60)
    print(f"  总数:{tot} 跳过:{skip} | 正确:{corr} | "
          f"Acc:{corr/tot*100:.2f}% | QWK:{qwk:.4f}")
    print("-"*60)
    for i,n in enumerate(cnames):
        if pt[i]>0:
            print(f"  {n:<14}: {pc[i]:>3}/{pt[i]:<3} = {pc[i]/pt[i]*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    TEST    = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    GT      = "C:/Users/Administrator/Desktop/PythonProject/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    CLS_MDL = "C:/Users/Administrator/Desktop/PythonProject/cls_output/best_cls_model_qwk.pth"
    SEG_MDL = "C:/Users/Administrator/Desktop/PythonProject/final_model_20260102_140556.pth"
    OUT     = "C:/Users/Administrator/Desktop/PythonProject/cls_output"
    DEV     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_predict_with_gt(TEST,GT,CLS_MDL,SEG_MDL,OUT,DEV,
                           image_size=512,seg_thresholds=[0.4,0.5,0.5,0.4])