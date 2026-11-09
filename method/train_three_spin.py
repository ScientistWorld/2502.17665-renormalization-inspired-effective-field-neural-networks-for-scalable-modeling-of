#!/usr/bin/env python3
"""Train EFNN or comparison networks on the binary three-spin task.

The trained network is exported as a TorchScript energy model (input: float32
spins of shape (N, 15) in {0, 1}; output: energies of shape (N,)), which is the
submission format the evaluators load. Model selection uses training loss only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.three_spin import load_binary
from method.models import build_model, count_parameters


class EnergyModel(nn.Module):
    """Network trained on standardized energies, returning raw energies."""

    def __init__(self, net: nn.Module, y_mean: float, y_std: float):
        super().__init__()
        self.net = net
        self.register_buffer("y_mean", torch.tensor(float(y_mean), dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(float(y_std), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.y_std + self.y_mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["binary"], default="binary")
    p.add_argument("--model", choices=["efnn", "dnn", "resnet", "densenet"], default="efnn")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--width", type=int, default=105)
    p.add_argument("--epochs", type=int, default=2400)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-gamma", type=float, default=0.997)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20250217, help="initialization and minibatch seed (data are fixed files)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--init-checkpoint", type=Path, default=None, help="start from saved weights (continued training)")
    p.add_argument("--output", type=Path, required=True, help="TorchScript energy model, e.g. outputs/binary_efnn.pt")
    p.add_argument("--checkpoint", type=Path, default=None, help="state_dict checkpoint to write")
    p.add_argument("--progress", type=Path, default=None, help="resumable training state (default derived from --checkpoint/--output)")
    return p.parse_args()


def run_signature(args: argparse.Namespace) -> str:
    keys = ["task", "model", "layers", "width", "epochs", "batch_size", "lr", "lr_gamma", "weight_decay", "seed", "init_checkpoint"]
    payload = json.dumps({k: str(getattr(args, k)) for k in keys}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def dev_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred.astype(np.float64) - y_true.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err * err)))
    return {
        "dev_rmse": rmse,
        "dev_relative_error": rmse / max(abs(float(np.mean(y_true))), 1e-12),
        "dev_mae": float(np.mean(np.abs(err))),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_binary()
    y_mean = float(np.mean(data["y_train"]))
    y_std = max(float(np.std(data["y_train"])), 1e-8)
    n_sites = data["x_train"].shape[1]
    print(
        f"device={device} model={args.model} layers={args.layers} width={args.width} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} gamma={args.lr_gamma} seed={args.seed} train_rows={len(data['y_train'])}",
        flush=True,
    )

    x_train = torch.from_numpy(data["x_train"]).float()
    y_train = torch.from_numpy((data["y_train"] - y_mean) / y_std).float()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = build_model(args.model, n_sites=n_sites, width=args.width, layers=args.layers).to(device)
    if args.init_checkpoint is not None:
        saved = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        if abs(saved["y_mean"] - y_mean) > 1e-6 or abs(saved["y_std"] - y_std) > 1e-6:
            raise SystemExit("init checkpoint was trained with a different target standardization")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)
    loss_fn = nn.MSELoss()

    if args.progress is not None:
        progress_path = args.progress
    elif args.checkpoint is not None:
        progress_path = args.checkpoint.with_suffix(".progress.pt")
    else:
        progress_path = args.output.with_suffix(".progress.ckpt")
    signature = run_signature(args)

    best_loss = float("inf")
    best_state = None
    first_epoch = 1
    if progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if progress.get("signature") == signature:
            model.load_state_dict(progress["model"])
            opt.load_state_dict(progress["optimizer"])
            sched.load_state_dict(progress["scheduler"])
            best_loss = progress["best_loss"]
            best_state = {k: v.cpu() for k, v in progress["best_state"].items()}
            torch.set_rng_state(progress["torch_rng"].cpu())
            np.random.set_state(progress["numpy_rng"])
            random.setstate(progress["python_rng"])
            first_epoch = progress["epoch"] + 1
            print(f"resuming at epoch={first_epoch} best_train_mse_z={best_loss:.6g}", flush=True)
        else:
            print("ignoring progress file from a different configuration", flush=True)

    for epoch in range(first_epoch, args.epochs + 1):
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
        if epoch % 50 == 0 and epoch < args.epochs:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            partial = progress_path.with_name(progress_path.name + ".partial")
            torch.save(
                {
                    "signature": signature,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                },
                partial,
            )
            os.replace(partial, progress_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    wrapper = EnergyModel(model, y_mean, y_std).eval()

    x_dev = torch.from_numpy(data["x_dev"]).float()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, x_dev[:8], check_inputs=[(x_dev[:5],), (x_dev[:33],)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    torch.jit.save(traced, str(tmp))
    os.replace(tmp, args.output)

    loaded = torch.jit.load(str(args.output), map_location="cpu")
    with torch.no_grad():
        eager = wrapper(x_dev).numpy().astype(np.float64)
        scripted = loaded(x_dev).numpy().astype(np.float64)
    gap = float(np.max(np.abs(eager - scripted)))
    if not gap < 1e-4:
        raise SystemExit(f"exported TorchScript model disagrees with the trained model (max gap {gap})")

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

    info = dev_metrics(data["y_dev"], scripted)
    info.update({"best_train_mse_z": best_loss, "param_count": float(count_parameters(model)), "export_gap": gap})
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "note": "Informational only; evaluators recompute every score from the model file.",
                "task": args.task,
                "model": args.model,
                "layers": args.layers,
                "width": args.width,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lr_gamma": args.lr_gamma,
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                "metrics": info,
            },
            indent=2,
        )
    )
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
