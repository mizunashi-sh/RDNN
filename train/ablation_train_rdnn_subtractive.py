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
from dataset import AngularVelocityDataset, SaccadeDataset
from model import RDNN


class AngularVelocityRDNN(nn.Module):
    def __init__(self, hidden_size, use_noise, noise_std, act_fn, rdnn_subtractive):
        super().__init__()
        self.init_mapper = nn.Linear(2, hidden_size)
        self.base_model = build_rdnn(
            input_size=1,
            hidden_size=hidden_size,
            output_size=2,
            use_noise=use_noise,
            noise_std=noise_std,
            act_fn=act_fn,
            rdnn_subtractive=rdnn_subtractive,
        )

    def forward(self, velocity, init_pos):
        hidden_0 = torch.relu(self.init_mapper(init_pos))
        return self.base_model(velocity, hidden_0)


@dataclass
class TrainConfig:
    task_name: str
    activation_name: str
    hidden_size: int
    rdnn_subtractive: bool
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


def parse_bool_token(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean token: {value}")


def get_activation(activation_name):
    activation_map = {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "rectified_tanh": rectified_tanh,
    }
    if activation_name not in activation_map:
        raise ValueError(f"Unknown activation: {activation_name}")
    return activation_map[activation_name]


def build_rdnn(input_size, hidden_size, output_size, use_noise, noise_std, act_fn, rdnn_subtractive):
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
        kwargs["subtractive"] = rdnn_subtractive
    elif "is_subtractive" in rdnn_signature.parameters:
        kwargs["is_subtractive"] = rdnn_subtractive
    return RDNN(**kwargs)


def build_model(config):
    act_fn = get_activation(config.activation_name)
    if config.task_name == "angular_velocity":
        return AngularVelocityRDNN(
            hidden_size=config.hidden_size,
            use_noise=config.use_noise,
            noise_std=config.noise_std,
            act_fn=act_fn,
            rdnn_subtractive=config.rdnn_subtractive,
        )
    if config.task_name == "saccade":
        return build_rdnn(
            input_size=3,
            hidden_size=config.hidden_size,
            output_size=2,
            use_noise=config.use_noise,
            noise_std=config.noise_std,
            act_fn=act_fn,
            rdnn_subtractive=config.rdnn_subtractive,
        )
    raise ValueError(f"Unknown task: {config.task_name}")


def circular_angular_error(preds, true_thetas):
    pred_angles = torch.remainder(torch.atan2(preds[..., 0], preds[..., 1]), 2 * math.pi)
    true_angles = torch.remainder(true_thetas, 2 * math.pi)
    angle_diff = torch.abs(pred_angles - true_angles)
    return torch.min(angle_diff, 2 * math.pi - angle_diff).mean()


def masked_mse(preds, targets, masks):
    return (masks * (preds - targets) ** 2).sum() / (masks.sum() * 2 + 1e-8)


def create_task_tensors(config, device):
    if config.task_name == "angular_velocity":
        dataset = AngularVelocityDataset(num_samples=config.train_samples, seq_len=config.seq_len, dt=config.dt)
        return {
            "inputs": dataset.velocities.to(device),
            "init_pos": dataset.init_pos.to(device),
            "targets": dataset.targets.to(device),
            "thetas": dataset.thetas.to(device),
        }

    if config.task_name == "saccade":
        dataset = SaccadeDataset(num_samples=config.train_samples, seq_len=config.seq_len)
        return {
            "inputs": dataset.inputs.to(device),
            "targets": dataset.targets.to(device),
            "masks": dataset.masks.to(device),
            "thetas": dataset.thetas.to(device),
        }

    raise ValueError(f"Unknown task: {config.task_name}")


def train_one_run(config, output_dir):
    torch.set_float32_matmul_precision("high")
    set_seed(config.seed)

    device = torch.device(config.device)
    model = build_model(config).to(device)

    if config.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    tensors = create_task_tensors(config, device)
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
            batch_inputs = tensors["inputs"][batch_idx]

            optimizer.zero_grad(set_to_none=True)
            if config.task_name == "angular_velocity":
                batch_init_pos = tensors["init_pos"][batch_idx]
                batch_targets = tensors["targets"][batch_idx]
                batch_thetas = tensors["thetas"][batch_idx]

                preds, _ = model(batch_inputs, batch_init_pos)
                loss = criterion(preds, batch_targets)
                batch_error = circular_angular_error(preds, batch_thetas)
            else:
                batch_targets = tensors["targets"][batch_idx]
                batch_masks = tensors["masks"][batch_idx]
                batch_thetas = tensors["thetas"][batch_idx]

                preds, _ = model(batch_inputs)
                loss = masked_mse(preds, batch_targets, batch_masks)
                batch_error = circular_angular_error(preds[:, -1, :], batch_thetas)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = batch_inputs.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_error += batch_error.item() * batch_size
            num_samples += batch_size

        avg_loss = epoch_loss / num_samples
        avg_error = epoch_error / num_samples
        history.append({"epoch": epoch, "train_loss": avg_loss, "angular_error": avg_error})
        print(
            f"[{config.task_name} | RDNN | subtractive={int(config.rdnn_subtractive)} | {config.activation_name} | "
            f"H={config.hidden_size} | seed={config.seed}] "
            f"epoch {epoch}/{config.epochs} - loss={avg_loss:.6f} angular_error={avg_error:.6f}"
        )

    run_dir = (
        output_dir
        / config.task_name
        / "RDNN"
        / f"subtractive_{int(config.rdnn_subtractive)}"
        / config.activation_name
        / f"hidden_{config.hidden_size}"
        / f"seed_{config.seed}"
    )
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
        "model_name": "RDNN",
        "final_train_loss": history[-1]["train_loss"],
        "best_train_loss": min(item["train_loss"] for item in history),
        "final_angular_error": history[-1]["angular_error"],
        "run_dir": str(run_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="RDNN subtractive-vs-divisive ablation for angular velocity and saccade tasks.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/contrast_train_rdnn_subtractive"))
    parser.add_argument("--tasks", nargs="*", default=["angular_velocity", "saccade"], choices=["angular_velocity", "saccade"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)

    parser.add_argument("--angular-train-samples", type=int, default=5000)
    parser.add_argument("--angular-seq-len", type=int, default=256)
    parser.add_argument("--angular-dt", type=float, default=0.1)

    parser.add_argument("--saccade-train-samples", type=int, default=5000)
    parser.add_argument("--saccade-seq-len", type=int, default=512)

    parser.add_argument("--use-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--rdnn-subtractive-options", nargs="*", default=["true"])
    parser.add_argument("--activations", nargs="*", default=["relu", "tanh", "rectified_tanh"])
    parser.add_argument("--hidden-sizes", nargs="*", type=int, default=[64, 128, 256])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    return parser.parse_args()


def build_config(args, task_name, activation_name, hidden_size, seed, rdnn_subtractive):
    if task_name == "angular_velocity":
        train_samples = args.angular_train_samples
        seq_len = args.angular_seq_len
        dt = args.angular_dt
    else:
        train_samples = args.saccade_train_samples
        seq_len = args.saccade_seq_len
        dt = 0.0

    return TrainConfig(
        task_name=task_name,
        activation_name=activation_name,
        hidden_size=hidden_size,
        rdnn_subtractive=rdnn_subtractive,
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_samples=train_samples,
        seq_len=seq_len,
        dt=dt,
        use_noise=args.use_noise,
        noise_std=args.noise_std,
        device=args.device,
        compile=args.compile,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subtractive_options = [parse_bool_token(v) for v in args.rdnn_subtractive_options]

    summaries = []
    for task_name, activation_name, hidden_size, seed, rdnn_subtractive in product(
        args.tasks,
        args.activations,
        args.hidden_sizes,
        args.seeds,
        subtractive_options,
    ):
        config = build_config(args, task_name, activation_name, hidden_size, seed, rdnn_subtractive)
        summaries.append(train_one_run(config, args.output_dir))

    with open(args.output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "task_name",
            "model_name",
            "activation_name",
            "hidden_size",
            "rdnn_subtractive",
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
