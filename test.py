#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

EPS = 1e-8

# ---------- 与训练一致的模型 ----------
class TinyVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16,16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(16,32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(32,64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 2)  # [g/r, b/r]

    def forward(self, x):
        return self.head(self.gap(self.features(x)).flatten(1))

# ---------- 与训练一致的预处理 ----------
def gray_world_wb_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    x = rgb_u8.astype(np.float32)
    means = x.reshape(-1, 3).mean(0) + EPS  # [R,G,B]
    gray  = means.mean()
    gains = gray / means
    y = x * gains[None, None, :]
    return np.clip(y, 0, 255).astype(np.uint8)

def compute_true_ratios(rgb_u8: np.ndarray):
    # 现在的真实值来自原图通道均值
    m = rgb_u8.reshape(-1, 3).mean(0).astype(np.float32) + EPS  # [R,G,B]
    R, G, B = m[0], m[1], m[2]
    gr_true = float(G / R)
    gb_true = float(G / B)
    return gr_true, gb_true

def load_image_as_input(path: Path, size: int = 32) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    orig = np.array(im)  # 原图（用于统计现在的 g/r, g/b）
    wb  = gray_world_wb_uint8(orig)  # 训练时输入是灰世界后的图
    wb32 = Image.fromarray(wb, "RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(wb32, dtype=np.float32) / 255.0      # [H,W,3]
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0) # [1,3,32,32]
    return x, orig

def percent_diff(pred: float, true: float) -> float:
    return abs(pred / (true + EPS) - 1.0) * 100.0

def main():
    ap = argparse.ArgumentParser(description="Infer g/r & g/b for one JPG with trained TinyVGG.")
    ap.add_argument("--image", type=str, default="iphone.jpg")
    ap.add_argument("--ckpt",  type=str, default="best.pt")
    args = ap.parse_args()

    img_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    assert img_path.exists(), f"找不到图片：{img_path}"
    assert ckpt_path.exists(), f"找不到模型：{ckpt_path}"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 1) 准备输入 + 计算当前真值
    x, orig = load_image_as_input(img_path, size=32)
    gr_true, gb_true = compute_true_ratios(orig)

    # 2) 加载模型
    model = TinyVGG().to(DEVICE)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        # 兼容直接保存 state_dict 的情况
        model.load_state_dict(state)
    model.eval()

    # 3) 前向推理（float32）
    with torch.no_grad():
        x = x.to(DEVICE)
        out = model(x).squeeze(0).cpu().numpy()  # [gr_pred, br_pred]
        gr_pred = float(out[0])
        br_pred = float(out[1])
        gb_pred = float(gr_pred / (br_pred + EPS))

    # 4) 误差统计
    gr_abs = abs(gr_pred - gr_true)
    gb_abs = abs(gb_pred - gb_true)
    gr_pct = percent_diff(gr_pred, gr_true)
    gb_pct = percent_diff(gb_pred, gb_true)

    # 5) 打印结果（仅核心信息）
    print(f"[IMAGE] {img_path.name}")
    print(f" now (from original):   g/r={gr_true:.6f}   g/b={gb_true:.6f}")
    print(f" pred (from model):     g/r={gr_pred:.6f}   g/b={gb_pred:.6f}")
    print(f" diff:                  |Δgr|={gr_abs:.6f}  |Δgb|={gb_abs:.6f}")
    print(f" percent diff:          gr={gr_pct:.3f}%    gb={gb_pct:.3f}%")

if __name__ == "__main__":
    main()
