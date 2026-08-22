import argparse
import csv
import inspect
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn

from activations import rectified_tanh
from dataset import DoubleAngularVelocityDataset
from model import RDNN, TaskGRU, TaskLSTM


class DoubleAngularVelocityNetwork(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.init_mapper = nn.Linear(4, hidden_size)
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
    if torch.cuda.is_available():
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


def build_rdnn(input_size, hidden_size, output_size, use_noise, noise_std, act_fn, subtractive):
    rdnn_signature = inspect.signature(RDNN.__init__)
    kwargs = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "use_noise": use_noise,
        "noise_std": noise_std,
        "act_fn": act_fn,
    }
    if "subtractive" in rdnn_signature.parameters:
        kwargs["subtractive"] = subtractive
    elif "is_subtractive" in rdnn_signature.parameters:
        kwargs["is_subtractive"] = subtractive
    return RDNN(**kwargs)


def build_model(model_name, hidden_size, activation_name, use_noise, noise_std):
    activation = get_activation(activation_name)
    if model_name == "RDNN":
        base_model = build_rdnn(
            input_size=2,
            hidden_size=hidden_size,
            output_size=4,
            use_noise=use_noise,
            noise_std=noise_std,
            act_fn=activation,
            subtractive=False,
        )
    elif model_name == "SubtractiveNetwork":
        base_model = build_rdnn(
            input_size=2,
            hidden_size=hidden_size,
            output_size=4,
            use_noise=use_noise,
            noise_std=noise_std,
            act_fn=activation,
            subtractive=True,
        )
    elif model_name == "TaskGRU":
        base_model = TaskGRU(
            input_size=2,
            hidden_size=hidden_size,
            output_size=4,
            use_noise=use_noise,
            noise_std=noise_std,
            act_fn=activation,
        )
    elif model_name == "TaskLSTM":
        base_model = TaskLSTM(
            input_size=2,
            hidden_size=hidden_size,
            output_size=4,
            use_noise=use_noise,
            noise_std=noise_std,
            act_fn=activation,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return DoubleAngularVelocityNetwork(base_model, hidden_size)


def angular_error(preds, true_thetas):
    pred_angles = torch.remainder(torch.atan2(preds[..., 0::2], preds[..., 1::2]), 2 * math.pi)
    true_angles = torch.remainder(true_thetas, 2 * math.pi)
    angle_diff = torch.abs(pred_angles - true_angles)
    return torch.min(angle_diff, 2 * math.pi - angle_diff).mean()


def train_one_run(config, output_dir):
    torch.set_float32_matmul_precision("high")
    set_seed(config.seed)

    device = torch.device(config.device)
    model = build_model(config.model_name, config.hidden_size, config.activation_name, config.use_noise, config.noise_std).to(device)

    if config.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    dataset = DoubleAngularVelocityDataset(num_samples=config.train_samples, seq_len=config.seq_len, dt=config.dt)
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
            batch_velocities = velocities[batch_idx]
            batch_init_pos = init_pos[batch_idx]
            batch_targets = targets[batch_idx]
            batch_thetas = thetas[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            preds, _ = model(batch_velocities, batch_init_pos)
            loss = criterion(preds, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = batch_velocities.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_error += angular_error(preds, batch_thetas).item() * batch_size
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
    parser = argparse.ArgumentParser(description="Train RDNN, TaskGRU, TaskLSTM, and SubtractiveNetwork on the double angular velocity task.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/contrast_train_double_angular_velocity"))
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
    parser.add_argument("--models", nargs="*", default=["RDNN", "TaskGRU", "TaskLSTM", "SubtractiveNetwork"])
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