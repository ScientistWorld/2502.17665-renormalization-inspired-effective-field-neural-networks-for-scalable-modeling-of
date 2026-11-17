#!/usr/bin/env python3
"""Train the symmetrized EFNN (or the DNN comparison) on the double-exchange task.

Trains on ``data/double_exchange`` split ``train`` (10 x 10 lattices) with the MSE of
standardized energies, Adam and an exponentially decaying learning rate, and keeps the
weights with the lowest training loss.

Initialization is selected on training data only: ``--restarts K`` random
initializations are each trained for the first ``--select-epochs W`` epochs of the
schedule, and the one with the lowest training loss in its W-th epoch is trained on to
the end of the same schedule (its optimizer, learning-rate and minibatch state carry
over, so the result is an ordinary full-length run of the selected initialization).
Candidate ``j`` uses seed ``seed + 1_000_000 * j``; candidate 0 is the plain ``--seed``
run. With ``--restarts 1`` or ``--init-checkpoint`` no selection happens.

The network is exported with ``torch.jit.script`` as a TorchScript energy model (input:
float32 spins of shape (N, 3, L, L), any L; output: energies of shape (N,)), which is
the submission format the evaluators load. Training is resumable from a progress file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.double_exchange import TRAIN_L, load_split
from method.models_de import EnergyModelDE, build_de_model, count_parameters

SEED_STRIDE = 1_000_000


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
    p.add_argument("--restarts", type=int, default=4, help="initializations tried; the best by training loss is kept")
    p.add_argument("--select-epochs", type=int, default=20, help="epochs each initialization trains before selection")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--train-limit", type=int, default=0, help="use only the first N training rows (smoke tests)")
    p.add_argument("--init-checkpoint", type=Path, default=None, help="start from saved weights (continued training)")
    p.add_argument("--output", type=Path, required=True, help="TorchScript energy model, e.g. outputs/de_efnn.pt")
    p.add_argument("--checkpoint", type=Path, default=None, help="state_dict checkpoint to write")
    p.add_argument("--progress", type=Path, default=None, help="resumable training state")
    return p.parse_args()


def run_signature(args: argparse.Namespace) -> str:
    keys = [
        "model", "channels", "layers", "kernel", "epochs", "batch_size", "lr", "lr_gamma", "seed",
        "restarts", "select_epochs", "train_limit", "init_checkpoint",
    ]
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


class Run:
    """One initialization with its optimizer, schedule, minibatch generator and best weights."""

    def __init__(self, args: argparse.Namespace, seed: int, site_mean: float, scale: float):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.net = build_de_model(args.model, args.channels, args.layers, args.kernel)
        if args.init_checkpoint is not None:
            saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            self.net.load_state_dict(saved["model"])
            if abs(saved["site_mean"] - site_mean) > 1e-9 or abs(saved["scale"] - scale) > 1e-9:
                raise SystemExit("init checkpoint was trained with a different target standardization")
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=args.lr_gamma)
        self.gen = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.last_loss = float("inf")
        self.best_loss = float("inf")
        self.best_state: dict | None = None

    def state(self) -> dict:
        return {
            "seed": self.seed,
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "generator": self.gen.get_state(),
            "epoch": self.epoch,
            "last_loss": self.last_loss,
            "best_loss": self.best_loss,
            "best_state": self.best_state,
        }

    def load(self, st: dict) -> None:
        self.net.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        self.gen.set_state(st["generator"])
        self.epoch = st["epoch"]
        self.last_loss = st["last_loss"]
        self.best_loss = st["best_loss"]
        self.best_state = st["best_state"]

    def train_epoch(self, x_train: torch.Tensor, z_train: torch.Tensor, batch_size: int, tag: str) -> None:
        t0 = time.time()
        self.net.train()
        n = x_train.shape[0]
        perm = torch.randperm(n, generator=self.gen)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx]
            zb = z_train[idx]
            self.opt.zero_grad(set_to_none=True)
            loss = torch.mean((self.net(xb) - zb) ** 2)
            loss.backward()
            self.opt.step()
            total += float(loss.detach()) * xb.shape[0]
        self.sched.step()
        self.epoch += 1
        avg = total / n
        self.last_loss = avg if math.isfinite(avg) else float("inf")
        if self.last_loss < self.best_loss:
            self.best_loss = self.last_loss
            self.best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        print(
            f"{tag}epoch={self.epoch} train_mse_z={avg:.6g} lr={self.sched.get_last_lr()[0]:.3g} "
            f"seconds={time.time() - t0:.1f}",
            flush=True,
        )


def save_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))

    x_np, y_np = load_split("train")
    if args.train_limit > 0:
        x_np, y_np = x_np[: args.train_limit], y_np[: args.train_limit]
    sites = TRAIN_L * TRAIN_L
    site_mean = float(np.mean(y_np)) / sites
    scale = max(float(np.std(y_np)), 1e-8)
    restarts = 1 if args.init_checkpoint is not None else max(1, args.restarts)
    select_epochs = min(max(1, args.select_epochs), args.epochs) if restarts > 1 else 0
    print(
        f"model={args.model} C={args.channels} layers={args.layers} kernel={args.kernel} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} gamma={args.lr_gamma} seed={args.seed} restarts={restarts} "
        f"select_epochs={select_epochs} train_rows={len(y_np)} site_mean={site_mean:.6f} scale={scale:.6f}",
        flush=True,
    )
    x_train = torch.from_numpy(x_np)
    z_train = torch.from_numpy((y_np - site_mean * sites) / scale).float()

    if args.progress is not None:
        progress_path = args.progress
    elif args.checkpoint is not None:
        progress_path = args.checkpoint.with_suffix(".progress.pt")
    else:
        progress_path = args.output.with_suffix(".progress.ckpt")
    signature = run_signature(args)

    progress: dict = {"signature": signature, "candidates": [], "selected": None, "run": None}
    if progress_path.exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=False)
        if saved.get("signature") == signature and "candidates" in saved:
            progress = saved
            print(
                f"resuming: {len(progress['candidates'])} candidate(s) done, "
                f"selected={progress['selected']}, main run epoch={progress['run']['epoch'] if progress['run'] else None}",
                flush=True,
            )
        else:
            print("ignoring progress file from a different configuration", flush=True)

    # Phase 1: selection among initializations (training data only).
    selection: list[dict] = []
    run: Run | None = None
    if progress["run"] is not None:
        run = Run(args, progress["run"]["seed"], site_mean, scale)
        run.load(progress["run"])
    elif restarts > 1:
        while len(progress["candidates"]) < restarts:
            j = len(progress["candidates"])
            cand = Run(args, args.seed + SEED_STRIDE * j, site_mean, scale)
            for _ in range(select_epochs):
                cand.train_epoch(x_train, z_train, args.batch_size, f"candidate={j} seed={cand.seed} ")
            progress["candidates"].append(cand.state())
            save_progress(progress_path, progress)
        losses = [c["last_loss"] for c in progress["candidates"]]
        best = int(np.argmin(losses))
        progress["selected"] = best
        print(f"selection after {select_epochs} epochs: losses={[f'{v:.6g}' for v in losses]} -> candidate {best}", flush=True)
        run = Run(args, progress["candidates"][best]["seed"], site_mean, scale)
        run.load(progress["candidates"][best])
    else:
        run = Run(args, args.seed, site_mean, scale)
    selection = [{"seed": c["seed"], "train_mse_z_after_selection_epochs": c["last_loss"]} for c in progress["candidates"]]

    # Phase 2: the selected initialization continues the same schedule to the end.
    while run.epoch < args.epochs:
        run.train_epoch(x_train, z_train, args.batch_size, "")
        if run.epoch % 5 == 0 and run.epoch < args.epochs:
            progress["run"] = run.state()
            save_progress(progress_path, progress)

    net = run.net
    best_loss = run.best_loss
    if run.best_state is not None:
        net.load_state_dict(run.best_state)
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
                "seed": run.seed,
                "base_seed": args.seed,
                "selection": selection,
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
                "selected_seed": run.seed,
                "selection": selection,
                "metrics": info,
            },
            indent=2,
        )
    )
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
