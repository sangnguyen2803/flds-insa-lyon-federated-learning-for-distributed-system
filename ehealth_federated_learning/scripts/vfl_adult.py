from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class PartyNet(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ServerHead(nn.Module):
    def __init__(self, emb_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([za, zb], dim=1))


@dataclass
class VFLConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    noise_std: float = 0.05
    emb_clip_norm: float = 5.0
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_fairness(
    y_true: np.ndarray, y_pred: np.ndarray, sex_values: np.ndarray
) -> dict[str, float]:
    group0 = sex_values == 0
    group1 = sex_values == 1

    def safe_rate(mask: np.ndarray, values: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        return float(values[mask].mean())

    positive_rate0 = safe_rate(group0, y_pred)
    positive_rate1 = safe_rate(group1, y_pred)
    dp_gap = abs(positive_rate0 - positive_rate1)

    tpr0_mask = group0 & (y_true == 1)
    tpr1_mask = group1 & (y_true == 1)
    tpr0 = safe_rate(tpr0_mask, y_pred == 1)
    tpr1 = safe_rate(tpr1_mask, y_pred == 1)
    eod_gap = abs(tpr0 - tpr1)

    acc0 = safe_rate(group0, (y_pred == y_true))
    acc1 = safe_rate(group1, (y_pred == y_true))
    acc_gap = abs(acc0 - acc1)

    return {
        "fairness_dp_gap": float(dp_gap),
        "fairness_eod_gap": float(eod_gap),
        "fairness_group_acc_gap": float(acc_gap),
    }


def load_data(csv_path: Path, seed: int):
    df = pd.read_csv(csv_path)
    features = [c for c in df.columns if c != "income"]
    sex_idx = features.index("sex")
    X = df[features].to_numpy(dtype=np.float32)
    y = df["income"].to_numpy(dtype=np.int64)
    sex = X[:, sex_idx].astype(np.int64)

    X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
        X, y, sex, test_size=0.2, stratify=y, random_state=seed
    )

    mid = X.shape[1] // 2
    Xa_train, Xb_train = X_train[:, :mid], X_train[:, mid:]
    Xa_test, Xb_test = X_test[:, :mid], X_test[:, mid:]

    train_ds = TensorDataset(
        torch.tensor(Xa_train),
        torch.tensor(Xb_train),
        torch.tensor(y_train),
    )
    test_ds = TensorDataset(
        torch.tensor(Xa_test),
        torch.tensor(Xb_test),
        torch.tensor(y_test),
    )
    return train_ds, test_ds, sex_test


def train_and_eval(cfg: VFLConfig, csv_path: Path) -> dict:
    set_seed(cfg.seed)
    train_ds, test_ds, sex_test = load_data(csv_path, cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    xa_dim = train_ds.tensors[0].shape[1]
    xb_dim = train_ds.tensors[1].shape[1]

    party_a = PartyNet(xa_dim)
    party_b = PartyNet(xb_dim)
    head = ServerHead()

    params = list(party_a.parameters()) + list(party_b.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    comm_bytes = 0
    emb_dim = 16
    for _ in range(cfg.epochs):
        party_a.train()
        party_b.train()
        head.train()
        for xa, xb, y in train_loader:
            optimizer.zero_grad()
            za = party_a(xa)
            zb = party_b(xb)

            za_norm = torch.norm(za, p=2, dim=1, keepdim=True).clamp(min=1e-6)
            zb_norm = torch.norm(zb, p=2, dim=1, keepdim=True).clamp(min=1e-6)
            za = za * (cfg.emb_clip_norm / za_norm).clamp(max=1.0)
            zb = zb * (cfg.emb_clip_norm / zb_norm).clamp(max=1.0)

            if cfg.noise_std > 0:
                za = za + torch.randn_like(za) * cfg.noise_std
                zb = zb + torch.randn_like(zb) * cfg.noise_std

            logits = head(za, zb)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch = xa.shape[0]
            emb_bytes = batch * emb_dim * 4
            comm_bytes += 4 * emb_bytes

    party_a.eval()
    party_b.eval()
    head.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xa, xb, y in test_loader:
            za = party_a(xa)
            zb = party_b(xb)
            logits = head(za, zb)
            pred = torch.argmax(logits, dim=1)
            y_true.append(y.numpy())
            y_pred.append(pred.numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    acc = float((y_true_np == y_pred_np).mean())
    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro"))
    fairness = compute_fairness(y_true_np, y_pred_np, sex_test)

    return {
        "method": "VFL_2party_adult",
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "privacy_noise_std": cfg.noise_std,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "estimated_comm_mb": float(comm_bytes / (1024**2)),
        **fairness,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/adult/adult.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--emb-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="runs/ehealth_summary/vfl_metrics.csv")
    args = parser.parse_args()

    cfg = VFLConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_std=args.noise_std,
        emb_clip_norm=args.emb_clip_norm,
        seed=args.seed,
    )
    results = train_and_eval(cfg, Path(args.csv))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(results)


if __name__ == "__main__":
    main()
