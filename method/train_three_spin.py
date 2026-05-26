#!/usr/bin/env python3
"""Train EFNN or comparison networks on the paper's three-spin tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.three_spin import ThreeSpinConfig, make_split, metric_bundle
from method.models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["binary", "continuous"], default="binary")
    p.add_argument("--model", choices=["efnn", "dnn", "resnet", "densenet"], default="efnn")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--width", type=int, default=105)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20250217)
    p.add_argument("--train-size", type=int, default=16000)
    p.add_argument("--test-size", type=int, default=4000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    return p.parse_args()


def standardize(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    std = max(std, 1e-8)
    return (y_train - mean) / std, (y_test - mean) / std, mean, std


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ThreeSpinConfig(seed=args.seed, train_size=args.train_size, test_size=args.test_size)
    split = make_split(args.task, cfg)
    y_train_z, _, y_mean, y_std = standardize(split["y_train"], split["y_test"])
    print(
        f"device={device} model={args.model} task={args.task} layers={args.layers} "
        f"width={args.width} epochs={args.epochs} train_size={args.train_size}",
        flush=True,
    )

    x_train = torch.from_numpy(split["x_train"]).float()
    y_train = torch.from_numpy(y_train_z).float()
    x_test = torch.from_numpy(split["x_test"]).float().to(device)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = build_model(args.model, n_sites=cfg.n_sites, width=args.width, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.985)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
        sched.step()
        avg = total / len(loader.dataset)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} train_mse_z={avg:.6g} lr={sched.get_last_lr()[0]:.3g}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_z = model(x_test).detach().cpu().numpy().astype(np.float32)
    y_pred = pred_z * y_std + y_mean

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        y_true=split["y_test"].astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        x_test=split["x_test"].astype(np.float32),
    )
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": args.model,
                "task": args.task,
                "layers": args.layers,
                "width": args.width,
                "seed": args.seed,
                "y_mean": y_mean,
                "y_std": y_std,
            },
            args.checkpoint,
        )

    metrics = metric_bundle(split["y_test"], y_pred, args.task)
    metrics["param_count"] = float(count_parameters(model))
    metrics["neuron_count"] = float(args.width)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "task": args.task,
                "model": args.model,
                "layers": args.layers,
                "width": args.width,
                "seed": args.seed,
                "epochs": args.epochs,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
