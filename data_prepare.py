#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置
IMG_EXT = ".jpg"  # 图像扩展名
IN_SIZE = 32      # 输入尺寸
CACHE_PATH = "data.npy"  # 缓存文件路径
EPS = 1e-8        # 避免除零
WORKERS = 16      # 并行处理线程数
MAX_RATIO = 4.0   # 比率最大值

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """将 sRGB [0,1] 转换为线性 RGB [0,1]"""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def gray_world_wb(rgb_linear: np.ndarray) -> np.ndarray:
    """在线性 RGB [0,1] 域应用灰世界白平衡, G 通道为锚，并规范化均值之和为 1"""
    means = rgb_linear.reshape(-1, 3).mean(axis=0) + EPS  # [R均值, G均值, B均值]
    gains = np.array([means[1] / means[0], 1.0, means[1] / means[2]], dtype=np.float32)
    wb = np.clip(rgb_linear * gains[None, None, :], 0, 1)
    # 规范化均值之和为 1
    wb_means = wb.reshape(-1, 3).mean(axis=0) + EPS
    k = 1.0 / (wb_means.sum() + EPS)  # 缩放系数
    wb = wb * k
    return wb

def compute_gt_ratios(rgb_linear: np.ndarray) -> tuple[float, float]:
    """在线性 RGB 均值基础上计算 G/R 和 G/B 真值比率, 限制最大值"""
    means = rgb_linear.reshape(-1, 3).mean(axis=0) + EPS
    gr = min(float(means[1] / means[0]), MAX_RATIO)  # 裁剪到 MAX_RATIO
    gb = min(float(means[1] / means[2]), MAX_RATIO)
    return gr, gb

def process_image(p: str) -> tuple[np.ndarray, np.ndarray]:
    """处理单张图像: 加载 uint8, 转为线性 RGB [0,1], 白平衡, 调整大小"""
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)  # 加载为 BGR uint8
    if bgr is None:
        raise ValueError(f"无法加载图像: {p}")
    rgb = bgr[:, :, ::-1]  # BGR 转 RGB
    rgb_linear = srgb_to_linear(rgb.astype(np.float32) / 255.0)  # 转为 [0,1] 并线性化
    gr, gb = compute_gt_ratios(rgb_linear)
    wb_linear = gray_world_wb(rgb_linear)
    wb_resized = cv2.resize(wb_linear, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_LINEAR)
    return wb_resized.transpose(2, 0, 1), np.array([gr, gb], dtype=np.float32)

def preload_parallel(root: str) -> tuple[np.ndarray, np.ndarray]:
    """并行预处理图像并保存为 npy 缓存"""
    files = sorted(glob(str(Path(root) / f"**/*{IMG_EXT}"), recursive=True))
    if not files:
        raise FileNotFoundError("未找到 JPG 文件")
    print(f"[INFO] 找到 {len(files)} 张图像, 使用 {WORKERS} 个线程预处理...")

    X = np.empty((len(files), 3, IN_SIZE, IN_SIZE), dtype=np.float32)
    Y = np.empty((len(files), 2), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(process_image, p) for p in files]
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(files), desc="预处理进度")):
            X[i], Y[i] = fut.result()

    print(f"[INFO] 预处理完成: X={X.shape} Y={Y.shape}")
    np.save(CACHE_PATH, {"X": X, "Y": Y}, allow_pickle=True)
    print(f"[INFO] 缓存保存至 {CACHE_PATH}")
    return X, Y

if __name__ == "__main__":
    preload_parallel("test_256")