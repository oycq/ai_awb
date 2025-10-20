#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from glob import glob
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# ========= 配置 =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 默认CUDA
torch.backends.cudnn.benchmark = True
torch.manual_seed(42); np.random.seed(42)
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXT   = ".jpg"
IN_SIZE   = 32
SAVE_PATH = "best.pt"
CACHE_FN  = "data.npy"
EPS = 1e-8

# ========= 灰世界（以 G 为锚） & 真值 =========
def gray_world_wb_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    """
    以 G 为锚：让 R、B 的均值被拉到等于 G 的均值
    R_gain = Ḡ / R̄,  G_gain = 1,  B_gain = Ḡ / B̄
    """
    x = rgb_u8.astype(np.float32)
    means = x.reshape(-1, 3).mean(0) + EPS  # [R̄, Ḡ, B̄]
    Rm, Gm, Bm = means[0], means[1], means[2]
    gains = np.array([Gm / Rm, 1.0, Gm / Bm], dtype=np.float32)
    y = x * gains[None, None, :]
    return np.clip(y, 0, 255).astype(np.uint8)

def compute_gt_ratios_from_original(rgb_u8: np.ndarray) -> Tuple[float, float]:
    """
    真值：g/r 与 g/b（注意不是 b/r）
    """
    m = rgb_u8.reshape(-1, 3).mean(0).astype(np.float32) + EPS  # [R̄, Ḡ, B̄]
    Rm, Gm, Bm = m[0], m[1], m[2]
    gr = Gm / Rm
    gb = Gm / Bm
    return float(gr), float(gb)

# ========= Tiny VGG（极简 + BN） =========
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
        self.head = nn.Linear(64, 2)  # 输出 [g/r, g/b]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.gap(self.features(x)).flatten(1))

# ========= 并行预加载 =========
def _load_one(i, p, in_size: int):
    im  = Image.open(p).convert("RGB")
    rgb = np.array(im)
    gr, gb = compute_gt_ratios_from_original(rgb)                    # 现在的真值: [g/r, g/b]
    wb  = gray_world_wb_uint8(rgb)                                   # G 锚灰世界
    wb  = Image.fromarray(wb, "RGB").resize((in_size, in_size), Image.BILINEAR)
    arr = (np.asarray(wb, dtype=np.float32) / 255.0).transpose(2,0,1)  # [3,H,W]
    return i, arr, np.array([gr, gb], dtype=np.float32)

def preload_mem_parallel(root: str, in_size=IN_SIZE, workers=8):
    root = Path(root)
    files = sorted(glob(str(root / f"**/*{IMG_EXT}"), recursive=True))
    files = [f for f in files if Path(f).is_file()]
    if not files:
        raise FileNotFoundError("没有找到 JPG 文件")
    N = len(files)
    print(f"[INFO] 找到 {N} 张 JPG，使用 {workers} 线程预加载...")

    X = np.empty((N, 3, in_size, in_size), dtype=np.float32)
    Y = np.empty((N, 2), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_load_one, i, p, in_size) for i, p in enumerate(files)]
        for fut in tqdm(as_completed(futures), total=len(futures), ncols=80, ascii=True, desc="预加载进度"):
            i, x_arr, y_arr = fut.result()
            X[i] = x_arr
            Y[i] = y_arr

    print(f"[INFO] 预加载完成 X={X.shape} Y={Y.shape} dtype=float32")
    return X, Y

# ========= 读取 / 写入缓存 =========
def save_cache(cache_path: Path, X: np.ndarray, Y: np.ndarray):
    np.save(cache_path, {"X": X, "Y": Y}, allow_pickle=True)

def load_cache(cache_path: Path):
    obj = np.load(cache_path, allow_pickle=True).item()
    return obj["X"], obj["Y"]

# ========= 评估：loss + MPE(gr) + MPE(gb) =========
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, lossf: nn.Module):
    model.eval()
    loss_sum, n = 0.0, 0
    mpe_gr_sum, mpe_gb_sum = 0.0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)         # [B,3,32,32]
        y = y.to(device, non_blocking=True)         # [B,2] -> [gr_true, gb_true]

        pred = model(x)                              # [B,2] -> [gr_pred, gb_pred]
        loss = lossf(pred, y)

        gr_t, gb_t = y[:, 0], y[:, 1]
        gr_p, gb_p = pred[:, 0], pred[:, 1]

        # MPE: mean( |pred/true - 1| )*100，避免除零
        mpe_gr = torch.mean(torch.abs(gr_p / (gr_t + EPS) - 1.0)) * 100.0
        mpe_gb = torch.mean(torch.abs(gb_p / (gb_t + EPS) - 1.0)) * 100.0

        bsz = x.size(0)
        loss_sum  += loss.item() * bsz
        mpe_gr_sum += mpe_gr.item() * bsz
        mpe_gb_sum += mpe_gb.item() * bsz
        n += bsz

    return loss_sum / n, mpe_gr_sum / n, mpe_gb_sum / n

# ========= 训练 =========
def train(
    data_root="val_256",
    epochs=30,
    batch_size=4096,
    lr=1e-3,
    val_ratio=0.1,
    test_ratio=0.0,          # 可选：划出测试集（默认不开）
    preload_workers=8,
    rebuild_cache=False,
):
    print(f"[INFO] 设备: {DEVICE}（float32 训练）")
    cache_path = Path(data_root) / CACHE_FN

    # 1) 数据准备：优先加载缓存
    if (not rebuild_cache) and cache_path.exists():
        print(f"[INFO] 发现缓存：{cache_path}，直接加载 ...")
        X_np, Y_np = load_cache(cache_path)
        print(f"[INFO] 缓存加载完成 X={X_np.shape} Y={Y_np.shape}")
    else:
        X_np, Y_np = preload_mem_parallel(data_root, IN_SIZE, workers=preload_workers)
        print(f"[INFO] 写入缓存：{cache_path}")
        save_cache(cache_path, X_np, Y_np)

    X = torch.from_numpy(X_np)  # float32
    Y = torch.from_numpy(Y_np)  # float32
    del X_np, Y_np

    # 2) 划分 train/val/(test)
    full = TensorDataset(X, Y)
    n_total = len(full)
    n_test  = int(n_total * test_ratio)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    assert n_train > 0 and n_val >= 1, "切分比例不合理"

    sets = list(random_split(full, [n_train, n_val, n_test] if n_test > 0 else [n_train, n_val]))
    train_set, val_set = sets[0], sets[1]
    test_set = sets[2] if (n_test > 0) else None

    print(f"[INFO] 训练集 {len(train_set)}，验证集 {len(val_set)}" + (f"，测试集 {len(test_set)}" if test_set else ""))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0) if test_set else None

    # 3) 模型 & 优化器 & 损失（MSE 回归）
    model = TinyVGG().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    best = float("inf")
    for e in range(1, epochs+1):
        # ---- Train ----
        model.train()
        tl_sum, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)      # y = [gr_true, gb_true]
            optim.zero_grad(set_to_none=True)
            pred = model(x)                          # pred = [gr_pred, gb_pred]
            loss = lossf(pred, y)
            loss.backward()
            optim.step()
            tl_sum += loss.item() * x.size(0)
            n += x.size(0)
        tr_loss = tl_sum / n

        # ---- Val (loss + mpe) ----
        val_loss, val_mpe_gr, val_mpe_gb = evaluate(model, val_loader, DEVICE, lossf)

        print(f"[EPOCH {e:03d}] train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}  "
              f"val_mpe_gr={val_mpe_gr:.3f}%  val_mpe_gb={val_mpe_gb:.3f}%")

        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "epoch": e, "val_loss": best}, SAVE_PATH)

    # ---- 可选 TEST ----
    if test_loader is not None:
        test_loss, test_mpe_gr, test_mpe_gb = evaluate(model, test_loader, DEVICE, lossf)
        print(f"[TEST] loss={test_loss:.6f}  mpe_gr={test_mpe_gr:.3f}%  mpe_gb={test_mpe_gb:.3f}%")

# ========= 入口 =========
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="val_256")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.0)   # 需要 test 时改 >0
    ap.add_argument("--preload_workers", type=int, default=8)
    ap.add_argument("--rebuild_cache", action="store_true")
    args = ap.parse_args()

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        preload_workers=args.preload_workers,
        rebuild_cache=args.rebuild_cache,
    )
