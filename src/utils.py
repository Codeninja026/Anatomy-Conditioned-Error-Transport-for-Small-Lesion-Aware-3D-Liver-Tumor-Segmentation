from __future__ import annotations
import json, random, os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=deterministic; torch.backends.cudnn.benchmark=not deterministic
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, history, config, batch=0, stale=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch":epoch,"batch":batch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"best":best,"stale":stale,"history":history,"config":config.__dict__}, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    d=torch.load(path,map_location=device); model.load_state_dict(d["model"])
    if optimizer: optimizer.load_state_dict(d["optimizer"])
    if scheduler: scheduler.load_state_dict(d["scheduler"])
    if scaler: scaler.load_state_dict(d["scaler"])
    return d

def metric_counts(logits, target):
    return metric_counts_from_prediction(logits.argmax(1), target)

def empty_metric_counts():
    return {key: 0 for key in (
        "liver_intersection", "liver_predicted", "liver_target", "liver_false_positive", "liver_false_negative",
        "tumor_intersection", "tumor_predicted", "tumor_target", "tumor_false_positive", "tumor_false_negative",
        "correct", "total")}

def metric_counts_from_prediction(pred, target):
    out={}
    for c,n in ((1,"liver"),(2,"tumor")):
        p=pred==c; y=target==c
        out[n+"_intersection"] = int((p&y).sum().item())
        out[n+"_predicted"] = int(p.sum().item())
        out[n+"_target"] = int(y.sum().item())
        out[n+"_false_positive"] = int((p & ~y).sum().item())
        out[n+"_false_negative"] = int((~p & y).sum().item())
    out["correct"] = int((pred==target).sum().item()); out["total"] = int(target.numel())
    return out

def summarize_metric_counts(counts):
    out={}
    for n in ("liver","tumor"):
        den=counts[n+"_predicted"]+counts[n+"_target"]
        out[n+"_dice"] = 2*counts[n+"_intersection"]/den if den else 1.0
        tp=counts[n+"_intersection"]; fp=counts[n+"_false_positive"]; fn=counts[n+"_false_negative"]
        out[n+"_precision"] = tp/max(tp+fp,1)
        out[n+"_recall"] = tp/max(tp+fn,1)
        out[n+"_iou"] = tp/max(tp+fp+fn,1)
    out["pixel_accuracy"] = counts["correct"]/max(counts["total"],1)
    return out

def metrics(logits, target):
    return summarize_metric_counts(metric_counts(logits, target))

def save_overlay(image, mask, pred, uncertainty, path, title):
    fig,ax=plt.subplots(1,5,figsize=(18,4)); ax[0].imshow(image[image.shape[0]//2],cmap="gray"); ax[0].set_title("CT")
    ax[1].imshow(image[image.shape[0]//2],cmap="gray"); ax[1].imshow(mask,alpha=.45,vmin=0,vmax=2); ax[1].set_title("Ground truth")
    ax[2].imshow(image[image.shape[0]//2],cmap="gray"); ax[2].imshow(pred,alpha=.45,vmin=0,vmax=2); ax[2].set_title("Prediction")
    ax[3].imshow(uncertainty,cmap="magma"); ax[3].set_title("Uncertainty")
    ax[4].imshow(image[image.shape[0]//2],cmap="gray"); ax[4].contour(pred==2,colors="red",linewidths=1); ax[4].set_title("Tumor boundary")
    for a in ax: a.axis("off")
    fig.suptitle(title); fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)
