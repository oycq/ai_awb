#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from model import TinyVGG  # 导入分离的 TinyVGG

# 配置
EPS = 1e-8              # 避免除零
IN_DIR = "input"        # 输入目录（RAW PNG）
OUT_DIR = "output"      # 输出目录（sRGB PNG）
CKPT_PATH = "best.pt"   # 模型检查点
BLACK_LEVEL = 9.125     # 黑电平
IN_SIZE = 32            # 网络输入尺寸
MAX_RATIO = 4.0         # 比率最大值（与 data_prep.py 一致）

# sRGB 到线性 RGB（与 data_prep.py 一致）
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """将 sRGB [0,1] 转换为线性 RGB [0,1]"""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

# 线性 RGB 到 sRGB（uint8）
def linear_to_srgb_uint8(bgr_linear_float_0_255: np.ndarray) -> np.ndarray:
    """将线性 RGB [0,255] 转换为 sRGB uint8"""
    x = bgr_linear_float_0_255.astype(np.float32) / 255.0
    mask = x <= 0.0031308
    x[mask] = x[mask] * 12.92
    x[~mask] = 1.055 * (x[~mask] ** (1.0 / 2.4)) - 0.055
    x = np.clip(x * 255.0, 0.0, 255.0)
    return x.astype(np.uint8)

# 灰世界白平衡（以 G 为锚，与 data_prep.py 一致）
def gray_world_G_anchor_rgb(rgb_linear: np.ndarray) -> np.ndarray:
    """在线性 RGB [0,1] 域应用灰世界白平衡, G 通道为锚，并规范化均值之和为 1"""
    means = rgb_linear.reshape(-1, 3).mean(axis=0) + EPS  # [R均值, G均值, B均值]
    gains = np.array([means[1] / means[0], 1.0, means[1] / means[2]], dtype=np.float32)
    wb = np.clip(rgb_linear * gains[None, None, :], 0, 1)
    # 规范化均值之和为 1
    wb_means = wb.reshape(-1, 3).mean(axis=0) + EPS
    k = 1.0 / (wb_means.sum() + EPS)  # 缩放系数
    wb = wb * k
    return wb

# 计算 G/R 和 G/B 比率（线性 RGB 域，裁剪到 MAX_RATIO）
def compute_gr_gb(rgb_linear: np.ndarray) -> tuple[float, float]:
    """在线性 RGB 均值基础上计算 G/R 和 G/B 比率, 裁剪到 MAX_RATIO"""
    means = rgb_linear.reshape(-1, 3).mean(axis=0) + EPS
    gr = min(float(means[1] / means[0]), MAX_RATIO)
    gb = min(float(means[1] / means[2]), MAX_RATIO)
    return gr, gb

# 准备网络输入（与 data_prep.py 一致）
def make_net_input_from_rgb(rgb_linear: np.ndarray, size: int = IN_SIZE) -> torch.Tensor:
    """从线性 RGB [0,1] 生成网络输入: 灰世界 -> resize -> 归一化 -> CHW"""
    wb = gray_world_G_anchor_rgb(rgb_linear)  # 线性 RGB 灰世界
    wb_resized = cv2.resize(wb, (size, size), interpolation=cv2.INTER_LINEAR)
    chw = wb_resized.astype(np.float32).transpose(2, 0, 1)  # [3,H,W], 确保 float32
    return torch.from_numpy(chw).unsqueeze(0)  # [1,3,H,W]

# 处理单张图像
def process_one(raw_path: Path, out_path: Path, model: TinyVGG, device: torch.device, black_level: float):
    """处理单张 RAW 图像: 去马赛克 -> 灰世界 -> 网络预测 -> 比率校正 -> sRGB"""
    # 1) 读取 RAW16（单通道 RGGB）
    raw16 = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)  # uint16, [H,W]
    if raw16 is None or raw16.ndim != 2:
        raise ValueError(f"期望单通道 16-bit Bayer PNG（RGGB）。错误文件：{raw_path}")

    # 2) RAW16 -> 线性 RGB [0,255]
    raw_f = raw16.astype(np.float32) / 256.0
    raw_f = np.maximum(raw_f - float(black_level), 0.0)
    raw_f = np.clip(raw_f, 0.0, 255.0)
    bayer8 = raw_f.astype(np.uint8)
    bgr_linear = cv2.demosaicing(bayer8, cv2.COLOR_BayerRG2BGR_VNG)  # 线性 BGR [0,255]
    rgb_linear = bgr_linear[..., ::-1]  # BGR -> RGB

    # 3) 计算当前比率（线性 RGB 域）
    rgb_linear_float = rgb_linear / 255.0  # [0,1]
    gr_now, gb_now = compute_gr_gb(rgb_linear_float)  # 裁剪到 MAX_RATIO

    # 4) 网络预测目标比率
    x = make_net_input_from_rgb(rgb_linear_float, size=IN_SIZE).to(device)
    with torch.no_grad():
        out = model(x).squeeze(0).cpu().numpy()
        gr_pred = float(out[0])  # 目标 G/R
        gb_pred = float(out[1])  # 目标 G/B

    # 5) 比率校正
    R_scale = float(gr_now / (gr_pred + EPS))
    B_scale = float(gb_now / (gb_pred + EPS))
    bgr_corr = np.empty_like(bgr_linear, dtype=np.float32)
    bgr_corr[..., 2] = np.clip(bgr_linear[..., 2].astype(np.float32) * R_scale, 0.0, 255.0)  # R
    bgr_corr[..., 1] = bgr_linear[..., 1].astype(np.float32)  # G 不动
    bgr_corr[..., 0] = np.clip(bgr_linear[..., 0].astype(np.float32) * B_scale, 0.0, 255.0)  # B

    # 6) 线性 -> sRGB，保存
    out_srgb = linear_to_srgb_uint8(bgr_corr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_srgb)

    # 7) 打印信息
    print(f"[IMAGE] {raw_path.name} -> {out_path.name}")
    print(f" now : G/R={gr_now:.6f}, G/B={gb_now:.6f}")
    print(f" pred: G/R={gr_pred:.6f}, G/B={gb_pred:.6f}")
    print(f" scales used: R_scale={R_scale:.6f}, B_scale={B_scale:.6f}")

def main():
    """批量处理 input 目录下的 RAW PNG，输出 sRGB PNG 到 output 目录"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 加载模型
    model = TinyVGG().to(device)
    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(f"找不到检查点文件：{CKPT_PATH}")
    state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    # 批量处理 input 目录
    in_dir = Path(IN_DIR)
    out_dir = Path(OUT_DIR)
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{IN_DIR}")
    pngs = sorted(in_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"{IN_DIR} 下没有找到 *.png 文件")

    print(f"[BATCH] 发现 {len(pngs)} 个 PNG，输出到：{OUT_DIR}")
    for raw_path in pngs:
        out_path = out_dir / raw_path.name
        process_one(raw_path, out_path, model, device, BLACK_LEVEL)

if __name__ == "__main__":
    main()