#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

EPS = 1e-8

# ----------------------------
# TinyVGG（与训练一致；输出 [g/r, g/b]）
# ----------------------------
class TinyVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16,16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 2)  # [g/r, g/b]

    def forward(self, x):
        return self.head(self.gap(self.features(x)).flatten(1))

# ----------------------------
# 灰世界（以 G 为锚，仅拉 R、B 到 G 的均值）
# R_gain = Ḡ/R̄,  G_gain = 1,  B_gain = Ḡ/B̄
# ----------------------------
def gray_world_G_anchor_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    x = rgb_u8.astype(np.float32)
    means = x.reshape(-1, 3).mean(0) + EPS  # [R̄,Ḡ,B̄]
    Rm, Gm, Bm = means[0], means[1], means[2]
    gains = np.array([Gm / Rm, 1.0, Gm / Bm], dtype=np.float32)
    y = x * gains[None, None, :]
    return np.clip(y, 0, 255).astype(np.uint8)

# ----------------------------
# sRGB gamma
# ----------------------------
def linear_to_srgb_uint8(bgr_linear_float_0_255: np.ndarray) -> np.ndarray:
    x = bgr_linear_float_0_255.astype(np.float32) / 255.0
    mask = x <= 0.0031308
    x[mask]  = x[mask] * 12.92
    x[~mask] = 1.055 * (x[~mask] ** (1.0 / 2.4)) - 0.055
    x = np.clip(x * 255.0, 0.0, 255.0)
    return x.astype(np.uint8)

# ----------------------------
# 统计全图 g/r 与 g/b（基于 RGB）
# ----------------------------
def compute_gr_gb(rgb_u8: np.ndarray):
    m = rgb_u8.reshape(-1, 3).astype(np.float32).mean(axis=0) + EPS  # [R̄,Ḡ,B̄]
    Rm, Gm, Bm = m[0], m[1], m[2]
    return float(Gm / Rm), float(Gm / Bm)  # g/r, g/b

# ----------------------------
# 准备网络输入（灰世界→resize→归一化；RGB→CHW）
# ----------------------------
def make_net_input_from_rgb(rgb_u8: np.ndarray, size: int = 32) -> torch.Tensor:
    wb = gray_world_G_anchor_rgb(rgb_u8)                      # G 锚灰世界
    wb32 = cv2.resize(wb, (size, size), interpolation=cv2.INTER_LINEAR)
    chw = (wb32.astype(np.float32) / 255.0).transpose(2, 0, 1)  # [3,H,W]
    return torch.from_numpy(chw).unsqueeze(0)  # [1,3,H,W]

def main():
    ap = argparse.ArgumentParser(description="Raw(16-bit, RGGB) -> demosaic -> grayworld(G-anchor) -> net -> ratio correction -> sRGB")
    ap.add_argument("--raw",  type=str, default="raw.png", help="single-channel 16-bit Bayer (RGGB) PNG")
    ap.add_argument("--ckpt", type=str, default="best.pt", help="trained checkpoint (TinyVGG)")
    ap.add_argument("--black", type=float, default=9.125, help="black level subtract after /256")
    ap.add_argument("--out",  type=str, default="output.png", help="output sRGB PNG")
    args = ap.parse_args()

    raw_path = Path(args.raw); ckpt_path = Path(args.ckpt)
    assert raw_path.exists(), f"找不到输入：{raw_path}"
    assert ckpt_path.exists(), f"找不到模型：{ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 1) RAW16 → /256 → 减黑电平 → clip
    raw16 = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)  # uint16, shape [H,W]
    if raw16 is None or raw16.ndim != 2:
        raise ValueError("期望单通道 16-bit Bayer PNG（RGGB），如 raw.png。")
    raw_f = raw16.astype(np.float32) / 256.0
    raw_f = np.maximum(raw_f - float(args.black), 0.0)
    raw_f = np.clip(raw_f, 0.0, 255.0)

    # 2) 去马赛克（RGGB），得到线性域 BGR(0..255)
    bayer8     = raw_f.astype(np.uint8)
    bgr_linear = cv2.demosaicing(bayer8, cv2.COLOR_BayerRG2BGR_VNG)  # 你也可换 EA/Linear
    rgb_linear = bgr_linear[..., ::-1]  # BGR -> RGB（仅用于统计与喂网络）

    # 3) 当前通道比值（未校正）
    gr_now, gb_now = compute_gr_gb(rgb_linear)

    # 4) 载入模型，计算“应该的” [g/r_pred, g/b_pred]
    model = TinyVGG().to(device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    x = make_net_input_from_rgb(rgb_linear, size=32).to(device)
    with torch.no_grad():
        out = model(x).squeeze(0).cpu().numpy()
        gr_pred = float(out[0])  # 目标 g/r
        gb_pred = float(out[1])  # 目标 g/b

    # 5) 把原始线性图的通道比值“恢复”为网络输出值
    #    目标：G/R -> gr_pred，G/B -> gb_pred
    #    调整尺度：R' = R * R_scale，B' = B * B_scale
    #    由于 G/R' = (G/R) / R_scale，因此取 R_scale = gr_now / gr_pred
    #          G/B' = (G/B) / B_scale，因此取 B_scale = gb_now / gb_pred
    R_scale = float(gr_now / (gr_pred + EPS))
    B_scale = float(gb_now / (gb_pred + EPS))

    bgr_corr = np.empty_like(bgr_linear, dtype=np.float32)
    bgr_corr[..., 2] = np.clip(bgr_linear[..., 2].astype(np.float32) * R_scale, 0.0, 255.0)  # R
    bgr_corr[..., 1] = bgr_linear[..., 1].astype(np.float32)                                  # G 不动
    bgr_corr[..., 0] = np.clip(bgr_linear[..., 0].astype(np.float32) * B_scale, 0.0, 255.0)  # B

    # 6) 线性 → sRGB，写文件
    out_srgb = linear_to_srgb_uint8(bgr_corr)
    cv2.imwrite(str(args.out), out_srgb)

    # 7) 简要打印
    print(f"[IMAGE] {raw_path.name} -> {args.out}")
    print(f" now : g/r={gr_now:.6f}  g/b={gb_now:.6f}")
    print(f" pred: g/r={gr_pred:.6f}  g/b={gb_pred:.6f}")
    print(f" scales used: R_scale={R_scale:.6f}  B_scale={B_scale:.6f}")

if __name__ == "__main__":
    main()
