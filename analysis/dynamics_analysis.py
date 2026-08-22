import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

from train.contrast_train_angular_velocity import build_model as build_av_model
from train.contrast_train_organics import build_model as build_organics_model
from train.contrast_train_saccade import build_model as build_sac_model
from train.contrast_train_neural_ode import build_model as build_neural_ode_model
from dataset import AngularVelocityDataset, SaccadeDataset
from model import RDNN, TaskGRU, TaskLSTM
from organics import ORGaNICs
from odenet import NeuralODE


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def disable_noise(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "use_noise"):
            m.use_noise = False
        if hasattr(m, "noise_std"):
            m.noise_std = 0.0


@dataclass
class DynamicsResult:
    task: str
    model_name: str
    activation_name: str
    hidden_size: int
    seed: int
    n_fixed_points: int
    n_stable_fixed_points: int
    n_saddle_points: int
    n_unstable_fixed_points: int
    n_marginal_fixed_points: int
    n_stable_limit_cycles: int
    n_uncertain_limit_cycles: int
    manifold_points: int
    manifold_bin_coverage: float
    manifold_flow_uniform_norm: float
    manifold_flow_median: float
    manifold_flow_p90: float
    manifold_flow_std: float
    manifold_reliable: bool


PAPER_NUM_SAMPLES = 1024
PAPER_SLOW_RATIO = 1e-3
PAPER_CONVERGENCE_TOL = 1e-4
PAPER_UNIQUENESS_TOL = 1e-2
ANALYSIS_CLIP_VALUE = 1e6


def sanitize_array(arr: np.ndarray, clip: float = ANALYSIS_CLIP_VALUE) -> np.ndarray:
    out = np.nan_to_num(arr, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(out, -clip, clip)


def safe_l2_norm(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    # Scale each vector before squaring to avoid overflow in x*x for very large values.
    arr = sanitize_array(arr)
    scale = np.max(np.abs(arr), axis=axis, keepdims=True)
    scale = np.where(scale < 1e-12, 1.0, scale)
    scaled = arr / scale
    norm = np.sqrt(np.sum(scaled * scaled, axis=axis))
    return np.squeeze(scale, axis=axis) * norm


class AutonomousDynamics:
    def __init__(self, model: torch.nn.Module, task: str):
        self.model = model
        self.task = task
        self.base = model.base_model if hasattr(model, "base_model") else model
        self.init_mapper = model.init_mapper if hasattr(model, "init_mapper") else None

        if isinstance(self.base, RDNN):
            self.kind = "RDNN"
        elif isinstance(self.base, TaskGRU):
            self.kind = "TaskGRU"
        elif isinstance(self.base, TaskLSTM):
            self.kind = "TaskLSTM"
        elif isinstance(self.base, NeuralODE):
            self.kind = "NeuralODE"
        elif isinstance(self.base, ORGaNICs):
            self.kind = "ORGaNICs"
        else:
            raise ValueError(f"Unsupported model type: {type(self.base)}")

        self.device = next(self.base.parameters()).device
        self.dtype = next(self.base.parameters()).dtype
        if self.kind == "NeuralODE":
            self.input_size = self.base.ode_func.net[0].in_features - self.base.hidden_size
        elif self.kind == "ORGaNICs":
            self.input_size = self.base.cell.input_size
        else:
            self.input_size = self.base.input_proj.in_features
        self.hidden_size = self.base.hidden_size if hasattr(self.base, "hidden_size") else self.base.cell.hidden_size

        with torch.no_grad():
            self.zero_input = torch.zeros(1, self.input_size, device=self.device, dtype=self.dtype)

    def _rdnn_step(self, state: torch.Tensor, input_t: torch.Tensor) -> torch.Tensor:
        R_t = state[:, : self.hidden_size]
        G_t = state[:, self.hidden_size :]

        I_t = self.base.input_proj(input_t)

        J = F.softplus(self.base.raw_J) if self.base.dale_principle else self.base.raw_J
        w = F.softplus(self.base.raw_w) if self.base.dale_principle else self.base.raw_w
        eta = F.softplus(self.base.raw_eta) + 1e-5 if self.base.dale_principle else self.base.raw_eta + 1e-5
        alpha_r = torch.sigmoid(self.base.raw_alpha_r)
        alpha_g = torch.sigmoid(self.base.raw_alpha_g)

        firing_rate = self.base.act_fn(R_t)
        recurrent_drive = F.linear(firing_rate, J)

        if self.base.static_gating:
            target_R = (recurrent_drive + I_t) / eta
        elif self.base.subtractive:
            target_R = recurrent_drive + I_t - G_t
        else:
            target_R = (recurrent_drive + I_t) / (eta + G_t)

        R_new = (1.0 - alpha_r) * R_t + alpha_r * target_R
        if self.base.dale_principle:
            R_new = F.relu(R_new)

        target_G = F.linear(firing_rate, w)
        G_new = (1.0 - alpha_g) * G_t + alpha_g * target_G
        if self.base.dale_principle:
            G_new = F.relu(G_new)

        return torch.cat([R_new, G_new], dim=-1)

    def _gru_step(self, state: torch.Tensor, input_t: torch.Tensor) -> torch.Tensor:
        hidden = state
        input_gates = self.base.input_proj(input_t)
        hidden_gates = self.base.hidden_proj(hidden)

        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)

        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        candidate = self.base.act_fn(input_new + reset_gate * hidden_new)
        return (1.0 - update_gate) * hidden + update_gate * candidate

    def _lstm_step(self, state: torch.Tensor, input_t: torch.Tensor) -> torch.Tensor:
        hidden = state[:, : self.hidden_size]
        cell = state[:, self.hidden_size :]

        input_gates = self.base.input_proj(input_t)
        hidden_gates = self.base.hidden_proj(hidden)

        input_gate, forget_gate, candidate_gate, output_gate = input_gates.chunk(4, dim=-1)
        hidden_input, hidden_forget, hidden_candidate, hidden_output = hidden_gates.chunk(4, dim=-1)

        input_gate = torch.sigmoid(input_gate + hidden_input)
        forget_gate = torch.sigmoid(forget_gate + hidden_forget)
        output_gate = torch.sigmoid(output_gate + hidden_output)

        candidate = self.base.act_fn(candidate_gate + hidden_candidate)
        cell_new = forget_gate * cell + input_gate * candidate
        hidden_new = output_gate * self.base.act_fn(cell_new)
        return torch.cat([hidden_new, cell_new], dim=-1)

    def _neural_ode_step(self, state: torch.Tensor, input_t: torch.Tensor) -> torch.Tensor:
        solver = odeint_adjoint if self.base.use_adjoint else odeint
        self.base.ode_func.x_t = input_t
        try:
            out = solver(self.base.ode_func, state, self.base.integration_time, method="rk4")
            return out[-1]
        finally:
            self.base.ode_func.x_t = None

    def _organics_step(self, state: torch.Tensor, input_t: torch.Tensor) -> torch.Tensor:
        y = state[:, : self.hidden_size]
        a = state[:, self.hidden_size : 2 * self.hidden_size]
        b0 = state[:, 2 * self.hidden_size : 3 * self.hidden_size]
        b1 = state[:, 3 * self.hidden_size :]

        y_new, a_new, b0_new, b1_new = self.base.cell(input_t, y, a, b0, b1)
        return torch.cat([y_new, a_new, b0_new, b1_new], dim=-1)

    def step(self, state: torch.Tensor, input_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_t is None:
            input_t = self.zero_input.repeat(state.shape[0], 1)

        if self.kind == "RDNN":
            return self._rdnn_step(state, input_t)
        if self.kind == "TaskGRU":
            return self._gru_step(state, input_t)
        if self.kind == "NeuralODE":
            return self._neural_ode_step(state, input_t)
        if self.kind == "ORGaNICs":
            return self._organics_step(state, input_t)
        return self._lstm_step(state, input_t)

    def decode_output(self, state: torch.Tensor) -> torch.Tensor:
        if self.kind == "RDNN":
            hidden = state[:, : self.hidden_size]
        elif self.kind == "TaskGRU":
            hidden = state
        elif self.kind == "NeuralODE":
            hidden = state
        elif self.kind == "ORGaNICs":
            hidden = state[:, : self.hidden_size]
        else:
            hidden = state[:, : self.hidden_size]
        if self.kind == "ORGaNICs":
            return self.base.fc(hidden)
        return self.base.fc_out(hidden)

    def init_state(self, batch_size: int, init_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        use_init = self.task == "angular_velocity" and self.init_mapper is not None and init_pos is not None

        if self.kind == "RDNN":
            if use_init:
                R0 = torch.relu(self.init_mapper(init_pos))
            else:
                if self.task == "angular_velocity" and self.init_mapper is not None and init_pos is None:
                    raise ValueError("Angular velocity requires init_pos when init_mapper is available")
                R0 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            G0 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            return torch.cat([R0, G0], dim=-1)

        if self.kind == "TaskGRU":
            if use_init:
                return torch.relu(self.init_mapper(init_pos))
            return torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)

        if self.kind == "NeuralODE":
            if use_init:
                return torch.relu(self.init_mapper(init_pos))
            return torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)

        if self.kind == "ORGaNICs":
            if use_init:
                y0 = torch.relu(self.init_mapper(init_pos))
            else:
                if self.task == "angular_velocity" and self.init_mapper is not None and init_pos is None:
                    raise ValueError("Angular velocity requires init_pos when init_mapper is available")
                y0 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            a0 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            b00 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            b10 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            return torch.cat([y0, a0, b00, b10], dim=-1)

        if use_init:
            h0 = torch.relu(self.init_mapper(init_pos))
        else:
            h0 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        c0 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        return torch.cat([h0, c0], dim=-1)



def load_trained_model(
    task: str,
    model_name: str,
    activation_name: str,
    hidden_size: int,
    seed: int,
    base_dir: Path,
    device: str = "cpu",
) -> torch.nn.Module:
    model_key = model_name.lower()
    ckpt_candidates = []
    if model_key == "neuralode":
        ckpt_candidates.extend(
            [
                base_dir / task / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
                base_dir / task / model_name / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
                base_dir / model_name / task / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
            ]
        )
    elif model_key == "organics":
        ckpt_candidates.extend(
            [
                base_dir / task / "organics" / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
                base_dir / task / model_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
                base_dir / task / model_name / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
            ]
        )
    else:
        ckpt_candidates.append(base_dir / task / model_name / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt")

    ckpt_path = next((path for path in ckpt_candidates if path.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found. Tried: {', '.join(str(path) for path in ckpt_candidates)}")

    if model_key == "neuralode":
        model = build_neural_ode_model(task, hidden_size, activation_name, use_noise=False, noise_std=0.0)
    elif model_key == "organics":
        model = build_organics_model(task, hidden_size, use_noise=False, noise_std=0.0)
    elif task == "angular_velocity":
        model = build_av_model(model_name, hidden_size, activation_name, use_noise=False, noise_std=0.0)
    elif task == "saccade":
        model = build_sac_model(model_name, hidden_size, activation_name, use_noise=False, noise_std=0.0)
    else:
        raise ValueError(f"Unknown task: {task}")

    model = model.to(device)

    try:
        data = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        data = torch.load(ckpt_path, map_location=device)

    state_dict = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    disable_noise(model)
    model.eval()
    return model


def collect_task_end_states(
    dyn: AutonomousDynamics,
    task: str,
    num_samples: int = 512,
    seq_len_av: int = 256,
    seq_len_sac: int = 512,
    dt: float = 0.1,
) -> torch.Tensor:
    with torch.no_grad():
        if task == "angular_velocity":
            ds = AngularVelocityDataset(num_samples=num_samples, seq_len=seq_len_av, dt=dt)
            velocities = ds.velocities.to(dyn.device, dtype=dyn.dtype)
            init_pos = ds.init_pos.to(dyn.device, dtype=dyn.dtype)
            state = dyn.init_state(num_samples, init_pos=init_pos)
            for t in range(velocities.shape[1]):
                state = dyn.step(state, velocities[:, t, :])
            return state

        ds = SaccadeDataset(num_samples=num_samples, seq_len=seq_len_sac)
        inputs = ds.inputs.to(dyn.device, dtype=dyn.dtype)
        state = dyn.init_state(num_samples)
        for t in range(inputs.shape[1]):
            state = dyn.step(state, inputs[:, t, :])
        return state


def simulate_autonomous(
    dyn: AutonomousDynamics,
    init_states: torch.Tensor,
    steps: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    states = [init_states]
    outputs = [dyn.decode_output(init_states)]
    state = init_states
    with torch.no_grad():
        for _ in range(steps):
            state = dyn.step(state)
            states.append(state)
            outputs.append(dyn.decode_output(state))
    return torch.stack(states, dim=1), torch.stack(outputs, dim=1)


def identify_ring_manifold(
    states: torch.Tensor,
    outputs: torch.Tensor,
    bins: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states_np = sanitize_array(states.detach().cpu().numpy())
    outputs_np = sanitize_array(outputs.detach().cpu().numpy())

    diff = outputs_np[:, 1:, :] - outputs_np[:, :-1, :]
    speed = safe_l2_norm(diff, axis=-1)

    selected_states = []
    selected_outputs = []
    selected_speed = []

    start_t = states_np.shape[1] // 4
    for i in range(states_np.shape[0]):
        traj_speed = speed[i]
        mask = np.zeros_like(traj_speed, dtype=bool)

        # Progressive relaxation to avoid empty manifold extraction on noisier solutions.
        max_speed = np.max(traj_speed[start_t:]) if traj_speed[start_t:].size > 0 else np.max(traj_speed)
        for ratio in [PAPER_SLOW_RATIO, 3e-3, 1e-2]:
            thr = max(1e-12, ratio * max_speed)
            mask[:] = False
            mask[start_t:] = traj_speed[start_t:] <= thr
            if mask.sum() >= 5:
                break

        if mask.sum() < 5:
            q = np.quantile(traj_speed[start_t:], 0.2)
            mask[:] = False
            mask[start_t:] = traj_speed[start_t:] <= q

        idx = np.where(mask)[0] + 1
        if idx.size == 0:
            continue

        selected_states.append(states_np[i, idx, :])
        selected_outputs.append(outputs_np[i, idx, :])
        selected_speed.append(speed[i, idx - 1])

    if not selected_states:
        raise RuntimeError("No slow points found; try longer autonomous simulation.")

    selected_states = np.concatenate(selected_states, axis=0)
    selected_outputs = np.concatenate(selected_outputs, axis=0)
    selected_speed = np.concatenate(selected_speed, axis=0)

    angles = np.arctan2(selected_outputs[:, 0], selected_outputs[:, 1])

    def pick_by_bins(n_bins: int):
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        ms, mo, ma = [], [], []
        for b in range(n_bins):
            left, right = edges[b], edges[b + 1]
            in_bin = (angles >= left) & (angles < right)
            idx = np.where(in_bin)[0]
            if idx.size == 0:
                continue
            best = idx[np.argmin(selected_speed[idx])]
            ms.append(selected_states[best])
            mo.append(selected_outputs[best])
            ma.append(angles[best])
        return ms, mo, ma

    manifold_states, manifold_outputs, manifold_angles = pick_by_bins(bins)
    if len(manifold_states) < max(16, bins // 8):
        manifold_states, manifold_outputs, manifold_angles = pick_by_bins(max(24, bins // 2))
    if len(manifold_states) < 12:
        manifold_states, manifold_outputs, manifold_angles = pick_by_bins(24)
    if len(manifold_states) < 8:
        # Final fallback: use all states (flatten across trajectories) and pick slowest per angular bin
        all_outputs = outputs_np.reshape((-1, outputs_np.shape[-1]))
        all_states = states_np.reshape((-1, states_np.shape[-1]))
        diff_all = (outputs_np[:, 1:, :] - outputs_np[:, :-1, :]).reshape((-1, outputs_np.shape[-1]))
        speed_all = safe_l2_norm(diff_all, axis=-1)
        n_traj, T, _ = outputs_np.shape
        # speed_all length == n_traj*(T-1); create a speed array aligned with flattened (n_traj*T)
        speed_all_full = np.concatenate([speed_all, np.full((n_traj,), speed_all.max() if speed_all.size > 0 else 1.0)])
        if speed_all_full.size < all_outputs.shape[0]:
            pad_size = all_outputs.shape[0] - speed_all_full.size
            speed_all_full = np.concatenate([speed_all_full, np.full((pad_size,), speed_all_full.max() if speed_all_full.size>0 else 1.0)])

        angles_all = np.arctan2(all_outputs[:, 0], all_outputs[:, 1])

        ms, mo, ma = [], [], []
        edges = np.linspace(-np.pi, np.pi, bins + 1)
        for b in range(bins):
            left, right = edges[b], edges[b + 1]
            in_bin = (angles_all >= left) & (angles_all < right)
            idx = np.where(in_bin)[0]
            if idx.size == 0:
                continue
            best_idx = idx[np.argmin(speed_all_full[idx])]
            ms.append(all_states[best_idx])
            mo.append(all_outputs[best_idx])
            ma.append(angles_all[best_idx])

        if len(ms) < 8:
            raise RuntimeError("Too few manifold points found even after fallback; model may not form a ring manifold.")

        manifold_states = np.array(ms)
        manifold_outputs = np.array(mo)
        manifold_angles = np.unwrap(np.array(ma))

    manifold_states = np.array(manifold_states)
    manifold_outputs = np.array(manifold_outputs)
    manifold_angles = np.unwrap(np.array(manifold_angles))

    order = np.argsort(manifold_angles)
    return manifold_states[order], manifold_outputs[order], manifold_angles[order]


def compute_flow_on_manifold(dyn: AutonomousDynamics, manifold_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = torch.tensor(manifold_states, device=dyn.device, dtype=dyn.dtype)
    with torch.no_grad():
        s_next = dyn.step(s)
        y = dyn.decode_output(s)
        y_next = dyn.decode_output(s_next)

    a = np.arctan2(y[:, 0].cpu().numpy(), y[:, 1].cpu().numpy())
    a_next = np.arctan2(y_next[:, 0].cpu().numpy(), y_next[:, 1].cpu().numpy())
    dtheta = wrap_angle(a_next - a)
    return a, dtheta


def detect_fixed_points_from_flow(angles: np.ndarray, dtheta: np.ndarray) -> List[Dict[str, float]]:
    points = []
    n = len(dtheta)
    for i in range(n):
        j = (i + 1) % n
        left = dtheta[i]
        right = dtheta[j]

        if left == 0.0:
            left = 1e-12
        if right == 0.0:
            right = -1e-12

        if np.sign(left) == np.sign(right):
            continue

        angle = 0.5 * (angles[i] + angles[j])
        if left > 0 and right < 0:
            fp_type = "stable"
        elif left < 0 and right > 0:
            fp_type = "saddle"
        else:
            fp_type = "unknown"

        points.append({"angle": float(angle), "index_left": int(i), "index_right": int(j), "flow_type": fp_type})

    return points


def jacobian_at_point(dyn: AutonomousDynamics, point: torch.Tensor) -> np.ndarray:
    point = point.detach().clone().requires_grad_(True)

    def f(z: torch.Tensor) -> torch.Tensor:
        return dyn.step(z.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(f, point)
    return J.detach().cpu().numpy()


def classify_fixed_point_from_jacobian(J: np.ndarray, eps: float = 1e-3) -> Tuple[str, np.ndarray]:
    eigvals = np.linalg.eigvals(J)
    mod = np.abs(eigvals)

    max_mod = float(np.max(mod))
    min_mod = float(np.min(mod))

    if max_mod < 1.0 - eps:
        return "stable", eigvals
    if min_mod > 1.0 + eps:
        return "unstable", eigvals
    if (mod > 1.0 + eps).any() and (mod < 1.0 - eps).any():
        return "saddle", eigvals
    return "marginal", eigvals


def unique_rows(points: List[np.ndarray], tol: float = 1e-2) -> List[np.ndarray]:
    unique: List[np.ndarray] = []
    for p in points:
        if not unique:
            unique.append(p)
            continue
        d = [float(safe_l2_norm(p - q, axis=-1)) for q in unique]
        if min(d) > tol:
            unique.append(p)
    return unique


def serialize_complex_spectrum(vals: np.ndarray) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for v in vals:
        out.append(
            {
                "real": float(np.real(v)),
                "imag": float(np.imag(v)),
                "abs": float(np.abs(v)),
                "angle": float(np.angle(v)),
            }
        )
    return out


def damping_ratios_from_discrete_eigs(vals: np.ndarray, dt: float = 1.0) -> List[Optional[float]]:
    # For discrete-time poles z, map to continuous-time lambda=log(z)/dt,
    # then zeta = -Re(lambda)/|lambda| for oscillatory modes.
    ratios: List[Optional[float]] = []
    for z in vals:
        if np.abs(z) < 1e-12:
            ratios.append(None)
            continue
        lam = np.log(z) / dt
        den = np.abs(lam)
        if den < 1e-12:
            ratios.append(None)
            continue
        ratios.append(float(-np.real(lam) / den))
    return ratios


def detect_limit_cycles(
    dyn: AutonomousDynamics,
    states: torch.Tensor,
    max_period: Optional[int] = None,
    tol: float = 5e-3,
    min_period: int = 2,
    min_repeats: int = 3,
    min_amplitude_ratio: float = 1e-3,
) -> List[Dict[str, object]]:
    states_np = sanitize_array(states.detach().cpu().numpy())
    cycles = []

    for i in range(states_np.shape[0]):
        traj = states_np[i]
        T = traj.shape[0]
        transient = T // 2
        tail = traj[transient:]

        # Auto-select period range from available tail length.
        # Need at least (min_repeats + 1) chunks to compare shifted segments.
        auto_max = tail.shape[0] // (min_repeats + 1)
        p_max = min(auto_max, max_period) if max_period is not None else auto_max
        if p_max < min_period:
            continue

        # Scale-aware tolerance: avoid rejecting long periods due to absolute-state magnitude.
        state_scale = float(np.mean(safe_l2_norm(tail, axis=1))) + 1e-12

        traj_candidates = []
        for p in range(min_period, p_max + 1):
            if tail.shape[0] < (min_repeats + 1) * p:
                continue

            # Compare adjacent period-length chunks near the trajectory end.
            # Example for min_repeats=3:
            # c0 = [-4p:-3p], c1 = [-3p:-2p], c2 = [-2p:-p], c3 = [-p:]
            chunks = [tail[-(k + 1) * p : -k * p if k > 0 else None] for k in range(min_repeats, -1, -1)]
            pair_diffs = []
            for k in range(len(chunks) - 1):
                pair_diffs.append(float(np.mean(safe_l2_norm(chunks[k] - chunks[k + 1], axis=1))))

            d = float(np.mean(pair_diffs))
            rel_d = d / state_scale
            if d < tol or rel_d < tol:
                cycle_pts = tail[-p:]
                amplitude = float(np.mean(safe_l2_norm(cycle_pts - cycle_pts.mean(axis=0, keepdims=True), axis=1)))
                if amplitude < min_amplitude_ratio * state_scale:
                    continue
                traj_candidates.append(
                    {
                        "period": p,
                        "points": cycle_pts,
                        "rel_error": rel_d,
                        "amplitude": amplitude,
                    }
                )

        if traj_candidates:
            # Prefer longer periods first (avoid collapsing long cycles to small harmonics).
            traj_candidates.sort(key=lambda c: (c["period"], -c["amplitude"], -c["rel_error"]))
            cycles.append(traj_candidates[-1])

    unique_cycles = []
    signatures = []
    for c in cycles:
        pts = c["points"]
        sig = float(np.mean(safe_l2_norm(pts - pts.mean(axis=0, keepdims=True), axis=1)))
        if len(signatures) == 0 or min(abs(sig - s) for s in signatures) > 1e-3:
            signatures.append(sig)
            unique_cycles.append(c)

    analyzed = []
    for c in unique_cycles:
        points = c["points"]
        M = np.eye(points.shape[1], dtype=np.float64)
        for k in range(points.shape[0]):
            p = torch.tensor(points[k], device=dyn.device, dtype=dyn.dtype)
            J = jacobian_at_point(dyn, p)
            M = J @ M

        eigvals = np.linalg.eigvals(M)
        max_mod = float(np.max(np.abs(eigvals)))
        stability = "stable" if max_mod < 1.0 else "uncertain"
        damping_ratios = damping_ratios_from_discrete_eigs(eigvals, dt=max(float(c["period"]), 1.0))
        analyzed.append(
            {
                "period": c["period"],
                "stability": stability,
                "max_multiplier": max_mod,
                "floquet_spectrum": serialize_complex_spectrum(eigvals),
                "damping_ratios": damping_ratios,
            }
        )

    return analyzed


def analyze_checkpoint(
    task: str,
    model_name: str,
    activation_name: str,
    hidden_size: int,
    seed: int,
    base_dir: str = "contrast_experiment_results",
    device: str = "cpu",
    num_samples: int = PAPER_NUM_SAMPLES,
    autonomous_steps: Optional[int] = None,
    manifold_bins: int = 256,
    cycle_max_period: Optional[int] = None,
) -> Dict[str, object]:
    base_path = Path(base_dir)
    model = load_trained_model(task, model_name, activation_name, hidden_size, seed, base_path, device=device)
    dyn = AutonomousDynamics(model, task)
    # If autonomous_steps not provided, follow paper: 32x task length (defaults in dataset)
    if autonomous_steps is None:
        if task == "angular_velocity":
            autonomous_steps = 256 * 32
        elif task == "saccade":
            autonomous_steps = 512 * 32
        else:
            autonomous_steps = 1024

    end_states = collect_task_end_states(dyn, task, num_samples=num_samples)
    states, outputs = simulate_autonomous(dyn, end_states, steps=autonomous_steps)

    manifold_states, manifold_outputs, manifold_angles = identify_ring_manifold(states, outputs, bins=manifold_bins)
    _, dtheta = compute_flow_on_manifold(dyn, manifold_states)
    fp_flow = detect_fixed_points_from_flow(manifold_angles, dtheta)

    # Uniform norm (L-infinity) of slow-manifold flow in output space, similar to reference vf_infty.
    manifold_outputs_t = torch.tensor(manifold_outputs, device=dyn.device, dtype=dyn.dtype)
    with torch.no_grad():
        manifold_outputs_next_t = dyn.decode_output(
            dyn.step(torch.tensor(manifold_states, device=dyn.device, dtype=dyn.dtype))
        )
    manifold_flow = (manifold_outputs_next_t - manifold_outputs_t).detach().cpu().numpy()
    manifold_flow_norms = safe_l2_norm(manifold_flow, axis=1)
    manifold_uniform_norm = float(np.max(manifold_flow_norms)) if manifold_flow_norms.size > 0 else float("nan")
    manifold_flow_median = float(np.median(manifold_flow_norms)) if manifold_flow_norms.size > 0 else float("nan")
    manifold_flow_p90 = float(np.percentile(manifold_flow_norms, 90)) if manifold_flow_norms.size > 0 else float("nan")
    manifold_flow_std = float(np.std(manifold_flow_norms)) if manifold_flow_norms.size > 0 else float("nan")
    manifold_bin_coverage = float(len(manifold_states) / max(1, manifold_bins))
    manifold_reliable = bool(
        len(manifold_states) >= max(16, manifold_bins // 8) and manifold_bin_coverage >= 0.25
    )

    fixed_points = []
    for item in fp_flow:
        idx = item["index_left"]
        state_guess = manifold_states[idx]
        p = torch.tensor(state_guess, device=dyn.device, dtype=dyn.dtype)
        J = jacobian_at_point(dyn, p)
        jac_type, eigvals = classify_fixed_point_from_jacobian(J)
        damping_ratios = damping_ratios_from_discrete_eigs(eigvals, dt=1.0)

        fixed_points.append(
            {
                "angle": item["angle"],
                "flow_type": item["flow_type"],
                "jacobian_type": jac_type,
                "max_abs_eig": float(np.max(np.abs(eigvals))),
                "jacobian_spectrum": serialize_complex_spectrum(eigvals),
                "damping_ratios": damping_ratios,
            }
        )

    # De-duplicate nearby fixed points in angle space.
    dedup = []
    for fp in fixed_points:
        if not dedup:
            dedup.append(fp)
            continue
        da = [abs(float(wrap_angle(np.array([fp["angle"] - x["angle"]]))[0])) for x in dedup]
        if min(da) > 0.05:
            dedup.append(fp)
    fixed_points = dedup

    if cycle_max_period is None:
        # For long-period oscillations (e.g. ~200 steps), allow broad search by default.
        cycle_max_period = 512

    cycles = detect_limit_cycles(dyn, states, max_period=cycle_max_period)

    count_stable = sum(1 for f in fixed_points if f["jacobian_type"] == "stable")
    count_saddle = sum(1 for f in fixed_points if f["jacobian_type"] == "saddle")
    count_unstable = sum(1 for f in fixed_points if f["jacobian_type"] == "unstable")
    count_marginal = sum(1 for f in fixed_points if f["jacobian_type"] == "marginal")

    count_cycle_stable = sum(1 for c in cycles if c["stability"] == "stable")
    count_cycle_uncertain = sum(1 for c in cycles if c["stability"] == "uncertain")

    summary = DynamicsResult(
        task=task,
        model_name=model_name,
        activation_name=activation_name,
        hidden_size=hidden_size,
        seed=seed,
        n_fixed_points=len(fixed_points),
        n_stable_fixed_points=count_stable,
        n_saddle_points=count_saddle,
        n_unstable_fixed_points=count_unstable,
        n_marginal_fixed_points=count_marginal,
        n_stable_limit_cycles=count_cycle_stable,
        n_uncertain_limit_cycles=count_cycle_uncertain,
        manifold_points=len(manifold_states),
        manifold_bin_coverage=manifold_bin_coverage,
        manifold_flow_uniform_norm=manifold_uniform_norm,
        manifold_flow_median=manifold_flow_median,
        manifold_flow_p90=manifold_flow_p90,
        manifold_flow_std=manifold_flow_std,
        manifold_reliable=manifold_reliable,
    )

    return {
        "summary": summary.__dict__,
        "fixed_points": fixed_points,
        "limit_cycles": cycles,
        "manifold_angles": manifold_angles.tolist(),
        "flow": dtheta.tolist(),
        "manifold_uniform_norm": manifold_uniform_norm,
        "manifold_flow_norms": manifold_flow_norms.tolist(),
        "manifold_flow_stats": {
            "bin_coverage": manifold_bin_coverage,
            "uniform_norm": manifold_uniform_norm,
            "median": manifold_flow_median,
            "p90": manifold_flow_p90,
            "std": manifold_flow_std,
            "reliable": manifold_reliable,
        },
    }


def analyze_batch(
    task: str,
    configs: List[Tuple[str, str, int, int]],
    base_dir: str = "contrast_experiment_results",
    device: str = "cpu",
    num_samples: int = 512,
    autonomous_steps: int = 1024,
    manifold_bins: int = 128,
) -> List[Dict[str, object]]:
    results = []
    for model_name, activation_name, hidden_size, seed in configs:
        out = analyze_checkpoint(
            task=task,
            model_name=model_name,
            activation_name=activation_name,
            hidden_size=hidden_size,
            seed=seed,
            base_dir=base_dir,
            device=device,
            num_samples=num_samples,
            autonomous_steps=autonomous_steps,
            manifold_bins=manifold_bins,
        )
        results.append(out)
    return results


def save_results_json(results: List[Dict[str, object]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
