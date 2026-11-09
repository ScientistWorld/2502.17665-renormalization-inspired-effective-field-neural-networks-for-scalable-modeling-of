#!/usr/bin/env python3
"""Train the symmetrized EFNN (or the DNN comparison) on the double-exchange task.

Trains on ``data/double_exchange`` split ``train`` (10 x 10 lattices) with the MSE of
standardized energies, Adam and an exponentially decaying learning rate, and keeps the
weights with the lowest training loss. The network is exported with
``torch.jit.script`` as a TorchScript energy model (input: float32 spins of shape
(N, 3, L, L), any L; output: energies of shape (N,)), which is the submission format
the evaluators load. Training is resumable from a progress file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.double_exchange import TRAIN_L, load_split
from method.models_de import EnergyModelDE, build_de_model, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["efnn", "dnn"], default="efnn")
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-gamma", type=float, default=0.97)
    p.add_argument("--seed", type=int, default=20250217, help="initialization and minibatch seed (data are fixed)")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--train-limit", type=int, default=0, help="use only the first N training rows (smoke tests)")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="start from saved weights (continued training)")
    p.add_argument("--output", type=Path, required=True, help="TorchScript energy model, e.g. outputs/de_efnn.pt")
    p.add_argument("--checkpoint", type=Path, default=None, help="state_dict checkpoint to write")
    p.add_argument("--progress", type=Path, default=None, help="resumable training state")
    return p.parse_args()


def run_signature(args: argparse.Namespace) -> str:
    keys = ["model", "channels", "layers", "kernel", "epochs", "batch_size", "lr", "lr_gamma", "seed", "train_limit", "init_checkpoint"]
    payload = json.dumps({k: str(getattr(args, k)) for k in keys}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred.astype(np.float64) - y_true.astype(np.float64)
    return float(np.sqrt(np.mean(err * err)) / abs(float(np.mean(y_true))))


def predict(model: nn.Module, x: np.ndarray, batch: int = 250) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            out.append(model(torch.from_numpy(x[i : i + batch])).double().numpy())
    return np.concatenate(out)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_np, y_np = load_split("train")
    if args.train_limit > 0:
        x_np, y_np = x_np[: args.train_limit], y_np[: args.train_limit]
    sites = TRAIN_L * TRAIN_L
    site_mean = float(np.mean(y_np)) / sites
    scale = max(float(np.std(y_np)), 1e-8)
    print(
        f"model={args.model} C={args.channels} layers={args.layers} kernel={args.kernel} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} gamma={args.lr_gamma} seed={args.seed} train_rows={len(y_np)} "
        f"site_mean={site_mean:.6f} scale={scale:.6f}",
        flush=True,
    )
    x_train = torch.from_numpy(x_np)
    z_train = torch.from_numpy((y_np - site_mean * sites) / scale).float()

    net = build_de_model(args.model, args.channels, args.layers, args.kernel)
    if args.init_checkpoint is not None:
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(saved["model"])
        if abs(saved["site_mean"] - site_mean) > 1e-9 or abs(saved["scale"] - scale) > 1e-9:
            raise SystemExit("init checkpoint was trained with a different target standardization")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)

    if args.progress is not None:
        progress_path = args.progress
    elif args.checkpoint is not None:
        progress_path = args.checkpoint.with_suffix(".progress.pt")
    else:
        progress_path = args.output.with_suffix(".progress.ckpt")
    signature = run_signature(args)
    gen = torch.Generator().manual_seed(args.seed)

    best_loss = float("inf")
    best_state = None
    first_epoch = 1
    if progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("signature") == signature:
            net.load_state_dict(progress["model"])
            opt.load_state_dict(progress["optimizer"])
            sched.load_state_dict(progress["scheduler"])
            best_loss = progress["best_loss"]
            best_state = progress["best_state"]
            gen.set_state(progress["generator"])
            first_epoch = progress["epoch"] + 1
            print(f"resuming at epoch={first_epoch} best_train_mse_z={best_loss:.6g}", flush=True)
        else:
            print("ignoring progress file from a different configuration", flush=True)

    n = x_train.shape[0]
    for epoch in range(first_epoch, args.epochs + 1):
        t0 = time.time()
        net.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb = x_train[idx]
            zb = z_train[idx]
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((net(xb) - zb) ** 2)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.shape[0]
        sched.step()
        avg = total / n
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"epoch={epoch} train_mse_z={avg:.6g} lr={sched.get_last_lr()[0]:.3g} seconds={time.time() - t0:.1f}", flush=True)
        if epoch % 5 == 0 and epoch < args.epochs:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            partial = progress_path.with_name(progress_path.name + ".partial")
            torch.save(
                {
                    "signature": signature,
                    "epoch": epoch,
                    "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "generator": gen.get_state(),
                },
                partial,
            )
            os.replace(partial, progress_path)

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    wrapper = EnergyModelDE(net, site_mean, scale).eval()
    scripted = torch.jit.script(wrapper)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    torch.jit.save(scripted, str(tmp))
    os.replace(tmp, args.output)

    loaded = torch.jit.load(str(args.output), map_location="cpu").eval()
    info: dict[str, float] = {"best_train_mse_z": best_loss, "param_count": float(count_parameters(net))}
    for split in ("dev_L10", "dev_L16", "dev_L24"):
        x_dev, y_dev = load_split(split)
        pred = predict(loaded, x_dev)
        if split == "dev_L10":
            with torch.no_grad():
                eager = wrapper(torch.from_numpy(x_dev[:64])).double().numpy()
            gap = float(np.max(np.abs(eager - pred[:64])))
            if not gap < 1e-3:
                raise SystemExit(f"exported TorchScript model disagrees with the trained model (max gap {gap})")
            info["export_gap"] = gap
        info[f"{split}_relative_error"] = relative_error(y_dev, pred)

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": net.state_dict(),
                "model_name": args.model,
                "channels": args.channels,
                "layers": args.layers,
                "kernel": args.kernel,
                "seed": args.seed,
                "site_mean": site_mean,
                "scale": scale,
            },
            args.checkpoint.with_name(args.checkpoint.name + ".tmp"),
        )
        os.replace(args.checkpoint.with_name(args.checkpoint.name + ".tmp"), args.checkpoint)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "note": "Informational only; evaluators recompute every score from the model file.",
                "task": "double_exchange",
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "metrics": info,
            },
            indent=2,
        )
    )
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
