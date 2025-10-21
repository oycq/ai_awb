#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from model import TinyVGG  # 导入分离的 TinyVGG

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU
CACHE_PATH = "data.npy"  # 缓存文件路径
SAVE_PATH = "best.pt"    # 模型保存路径
EPOCHS = 100             # 训练轮数
BATCH_SIZE = 2048        # 批次大小
VAL_RATIO = 0.1          # 验证集比例
EPS = 1e-8               # 避免除零
WEIGHT_DECAY = 1e-5      # L2 正则化

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)

def clipped_relative_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """自定义损失: 比例差裁剪到 3"""
    rel_diff = torch.abs((pred - true) / (true + EPS))
    clipped_diff = torch.clamp(rel_diff, max=3.0)
    return torch.mean(clipped_diff)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float, float]:
    """评估模型, 返回损失和 MPE 指标"""
    model.eval()
    loss_sum, mpe_gr_sum, mpe_gb_sum, n = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        pred = model(x)
        loss = clipped_relative_loss(pred, y)
        gr_p, gb_p = pred[:, 0], pred[:, 1]
        gr_t, gb_t = y[:, 0], y[:, 1]
        mpe_gr = torch.mean(torch.abs(gr_p / (gr_t + EPS) - 1)) * 100
        mpe_gb = torch.mean(torch.abs(gb_p / (gb_t + EPS) - 1)) * 100
        loss_sum += loss.item() * x.size(0)
        mpe_gr_sum += mpe_gr.item() * x.size(0)
        mpe_gb_sum += mpe_gb.item() * x.size(0)
        n += x.size(0)
    return loss_sum / n, mpe_gr_sum / n, mpe_gb_sum / n

def train_model():
    """训练模型, 使用缓存数据并保存最佳模型"""
    print(f"[INFO] 设备: {DEVICE}")
    if not Path(CACHE_PATH).exists():
        raise FileNotFoundError(f"未找到缓存文件 {CACHE_PATH}, 请先运行 data_prep.py")

    data = np.load(CACHE_PATH, allow_pickle=True).item()
    X, Y = torch.from_numpy(data["X"]), torch.from_numpy(data["Y"])

    dataset = TensorDataset(X, Y)
    n_total = len(dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"[INFO] 训练集: {n_train}, 验证集: {n_val}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, pin_memory=True)

    model = TinyVGG().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = clipped_relative_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = train_loss_sum / n

        val_loss, val_mpe_gr, val_mpe_gb = evaluate(model, val_loader)
        print(f"[轮次 {epoch:03d}] 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | "
              f"MPE GR: {val_mpe_gr:.3f}% | MPE GB: {val_mpe_gb:.3f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, SAVE_PATH)


if __name__ == "__main__":
    train_model()