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
from torch.utils.data import DataLoader, Dataset

from activations import rectified_tanh
from model import RDNN, TaskGRU, TaskLSTM
from dataset import AngularVelocityDataset

class AngularVelocityNetwork(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.init_mapper = nn.Linear(2, hidden_size)
        self.base_model = base_model

    def forward(self, velocity, init_pos):
        hidden_0 = torch.relu(self.init_mapper(init_pos))
        return self.base_model(velocity, hidden_0)


@dataclass
class TrainConfig:
    model_name: str
    activation_name: str
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
    torch.cuda.manual_seed_all(seed)


def get_activation(activation_name):
    activation_map = {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "rectified_tanh": rectified_tanh,
    }
    if activation_name not in activation_map:
        raise ValueError(f"Unknown activation: {activation_name}")
    return activation_map[activation_name]


def build_model(model_name, hidden_size, activation_name, use_noise, noise_std):
    activation = get_activation(activation_name)
    base_kwargs = dict(input_size=1, hidden_size=hidden_size, output_size=2, use_noise=use_noise, noise_std=noise_std, act_fn=activation)

    if model_name == "RDNN":
        base_model = RDNN(**base_kwargs)
    elif model_name == "TaskGRU":
        base_model = TaskGRU(**base_kwargs)
    elif model_name == "TaskLSTM":
        base_model = TaskLSTM(**base_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return AngularVelocityNetwork(base_model, hidden_size)


def angular_error(preds, true_thetas):
    pred_angles = torch.remainder(torch.atan2(preds[..., 0], preds[..., 1]), 2 * math.pi)
    true_angles = torch.remainder(true_thetas, 2 * math.pi)
    angle_diff = torch.abs(pred_angles - true_angles)
    return torch.min(angle_diff, 2 * math.pi - angle_diff).mean()


def train_one_run(config, output_dir):
    torch.set_float32_matmul_precision('high')
    set_seed(config.seed)

    device = torch.device(config.device)
    model = build_model(config.model_name, config.hidden_size, config.activation_name, config.use_noise, config.noise_std).to(device)

    if config.compile:
        import platform
        if hasattr(torch, "compile") and platform.system() != "Windows":
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
        else:
            print("Warning: torch.compile not supported in this environment.")

    dataset = AngularVelocityDataset(num_samples=config.train_samples, seq_len=config.seq_len, dt=config.dt)
    
    # 将整个数据集提前放到设备上，避免 DataLoader 中每个 batch 的 CPU->GPU 通信开销
    dataset.velocities = dataset.velocities.to(device)
    dataset.init_pos = dataset.init_pos.to(device)
    dataset.targets = dataset.targets.to(device)
    dataset.thetas = dataset.thetas.to(device)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
        
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    history = []
    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_error = 0.0
        num_samples = 0

        for velocities, init_pos, targets, true_thetas in loader:
            # velocities, etc. 已经预加载到 GPU，无需再调用 .to(device)

            optimizer.zero_grad(set_to_none=True)
            preds, _ = model(velocities, init_pos)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = velocities.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_error += angular_error(preds, true_thetas).item() * batch_size
            num_samples += batch_size

        avg_loss = epoch_loss / num_samples
        avg_error = epoch_error / num_samples
        history.append({"epoch": epoch, "train_loss": avg_loss, "angular_error": avg_error})
        print(
            f"[{config.model_name} | {config.activation_name} | H={config.hidden_size} | seed={config.seed}] "
            f"epoch {epoch}/{config.epochs} - loss={avg_loss:.6f} angular_error={avg_error:.6f}"
        )

    run_dir = output_dir / config.model_name / config.activation_name / f"hidden_{config.hidden_size}" / f"seed_{config.seed}"
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
        "final_train_loss": history[-1]["train_loss"],
        "best_train_loss": min(item["train_loss"] for item in history),
        "final_angular_error": history[-1]["angular_error"],
        "run_dir": str(run_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train RDNN, TaskGRU, and TaskLSTM on the angular velocity task.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/contrast_train_angular_velocity"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--use-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Enable torch.compile for faster execution on modern GPUs.")
    parser.add_argument("--models", nargs="*", default=["RDNN", "TaskGRU", "TaskLSTM"])
    parser.add_argument("--activations", nargs="*", default=["relu", "tanh"])
    parser.add_argument("--hidden-sizes", nargs="*", type=int, default=[64, 128, 256])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(5)))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name, activation_name, hidden_size, seed in product(args.models, args.activations, args.hidden_sizes, args.seeds):
        if model_name == "TaskLSTM" and activation_name == "relu":
            continue

        config = TrainConfig(
            model_name=model_name,
            activation_name=activation_name,
            hidden_size=hidden_size,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_samples=args.train_samples,
            seq_len=args.seq_len,
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
            "activation_name",
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
