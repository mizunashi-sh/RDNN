import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn

from dataset import AngularVelocityDataset, SaccadeDataset
from organics import ORGaNICs


class AngularVelocityORGaNICs(nn.Module):
    def __init__(self, hidden_size, use_noise, noise_std):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_mapper = nn.Linear(2, hidden_size)
        self.base_model = ORGaNICs(
            input_size=1,
            hidden_size=hidden_size,
            output_size=2,
            use_noise=use_noise,
            noise_std=noise_std,
        )

    def forward(self, velocity, init_pos):
        y0 = torch.relu(self.init_mapper(init_pos))
        batch_size = velocity.size(0)
        device = velocity.device
        dtype = velocity.dtype
        a0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        b00 = torch.zeros_like(a0)
        b10 = torch.zeros_like(a0)
        return self.base_model(velocity, y=y0, a=a0, b0=b00, b1=b10)


@dataclass
class TrainConfig:
    task_name: str
    hidden_size: int
    seed: int
    epochs: int
    batch_size: int
    lr: float
    train_samples: int
    seq_len: int
    dt: float
    use_noise: bool
    noise_std: float
    device: str
    compile: bool


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(task_name, hidden_size, use_noise, noise_std):
    if task_name == "angular_velocity":
        return AngularVelocityORGaNICs(hidden_size=hidden_size, use_noise=use_noise, noise_std=noise_std)

    if task_name == "saccade":
        return ORGaNICs(
            input_size=3,
            hidden_size=hidden_size,
            output_size=2,
            use_noise=use_noise,
            noise_std=noise_std,
        )

    raise ValueError(f"Unknown task: {task_name}")


def build_dataset(task_name, train_samples, seq_len, dt):
    if task_name == "angular_velocity":
        return AngularVelocityDataset(num_samples=train_samples, seq_len=seq_len, dt=dt)

    if task_name == "saccade":
        return SaccadeDataset(num_samples=train_samples, seq_len=seq_len)

    raise ValueError(f"Unknown task: {task_name}")


def angular_error(preds, true_thetas):
    pred_angles = torch.remainder(torch.atan2(preds[..., 0], preds[..., 1]), 2 * math.pi)
    true_angles = torch.remainder(true_thetas, 2 * math.pi)
    angle_diff = torch.abs(pred_angles - true_angles)
    return torch.min(angle_diff, 2 * math.pi - angle_diff).mean()


def masked_mse(preds, targets, masks):
    return (masks * (preds - targets) ** 2).sum() / (masks.sum() * 2 + 1e-8)


def train_one_run(config, output_dir):
    torch.set_float32_matmul_precision("high")
    set_seed(config.seed)

    device = torch.device(config.device)
    model = build_model(
        task_name=config.task_name,
        hidden_size=config.hidden_size,
        use_noise=config.use_noise,
        noise_std=config.noise_std,
    ).to(device)

    if config.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    dataset = build_dataset(config.task_name, config.train_samples, config.seq_len, config.dt)

    if config.task_name == "saccade":
        inputs = dataset.inputs.to(device)
        targets = dataset.targets.to(device)
        masks = dataset.masks.to(device)
        thetas = dataset.thetas.to(device)
    else:
        velocities = dataset.velocities.to(device)
        init_pos = dataset.init_pos.to(device)
        targets = dataset.targets.to(device)
        thetas = dataset.thetas.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    history = []
    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_error = 0.0
        num_samples = 0

        perm = torch.randperm(config.train_samples, device=device)
        for start in range(0, config.train_samples, config.batch_size):
            batch_idx = perm[start:start + config.batch_size]

            optimizer.zero_grad(set_to_none=True)

            if config.task_name == "saccade":
                batch_inputs = inputs[batch_idx]
                batch_targets = targets[batch_idx]
                batch_masks = masks[batch_idx]
                batch_thetas = thetas[batch_idx]

                preds, _ = model(batch_inputs)
                loss = masked_mse(preds, batch_targets, batch_masks)
                batch_size = batch_inputs.size(0)
                batch_error = angular_error(preds[:, -1, :], batch_thetas)
            else:
                batch_velocities = velocities[batch_idx]
                batch_init_pos = init_pos[batch_idx]
                batch_targets = targets[batch_idx]
                batch_thetas = thetas[batch_idx]

                preds, _ = model(batch_velocities, batch_init_pos)
                loss = criterion(preds, batch_targets)
                batch_size = batch_velocities.size(0)
                batch_error = angular_error(preds, batch_thetas)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            epoch_error += batch_error.item() * batch_size
            num_samples += batch_size

        avg_loss = epoch_loss / num_samples
        avg_error = epoch_error / num_samples
        history.append({"epoch": epoch, "train_loss": avg_loss, "angular_error": avg_error})
        print(
            f"[ORGaNICs | {config.task_name} | H={config.hidden_size} | seed={config.seed}] "
            f"epoch {epoch}/{config.epochs} - loss={avg_loss:.6f} angular_error={avg_error:.6f}"
        )

    run_dir = output_dir / config.task_name / "organics" / f"hidden_{config.hidden_size}" / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "angular_error"])
        writer.writeheader()
        writer.writerows(history)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        run_dir / "checkpoint.pt",
    )

    return {
        **asdict(config),
        "model_name": "ORGaNICs",
        "final_train_loss": history[-1]["train_loss"],
        "best_train_loss": min(item["train_loss"] for item in history),
        "final_angular_error": history[-1]["angular_error"],
        "run_dir": str(run_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train ORGaNICs on angular velocity and memory-guided saccade tasks.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/contrast_train_organics"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=256, help="Default seq_len for angular velocity. Saccade uses --saccade-seq-len.")
    parser.add_argument("--saccade-seq-len", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--use-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Enable torch.compile for faster execution on modern GPUs.")
    parser.add_argument("--tasks", nargs="*", default=["angular_velocity", "saccade"])
    parser.add_argument("--hidden-sizes", nargs="*", type=int, default=[64, 128, 256])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(5)))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for task_name, hidden_size, seed in product(args.tasks, args.hidden_sizes, args.seeds):
        task_seq_len = args.saccade_seq_len if task_name == "saccade" else args.seq_len

        config = TrainConfig(
            task_name=task_name,
            hidden_size=hidden_size,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_samples=args.train_samples,
            seq_len=task_seq_len,
            dt=args.dt,
            use_noise=args.use_noise,
            noise_std=args.noise_std,
            device=args.device,
            compile=args.compile,
        )
        summaries.append(train_one_run(config, args.output_dir))

    with open(args.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_name",
            "task_name",
            "hidden_size",
            "seed",
            "epochs",
            "batch_size",
            "lr",
            "train_samples",
            "seq_len",
            "dt",
            "use_noise",
            "noise_std",
            "device",
            "compile",
            "final_train_loss",
            "best_train_loss",
            "final_angular_error",
            "run_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
