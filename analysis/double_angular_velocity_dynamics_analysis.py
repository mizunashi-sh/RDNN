import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

from train.contrast_train_double_angular_velocity import build_model as build_double_av_model
from train.contrast_train_neural_ode import build_model as build_neural_ode_model
from dataset import DoubleAngularVelocityDataset
from model import RDNN, TaskGRU, TaskLSTM
from odenet import NeuralODE


TASK_NAME = "double_angular_velocity"
PAPER_NUM_SAMPLES = 1024
PAPER_SLOW_RATIO = 1e-3
PAPER_MANIFOLD_BINS = 32


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def disable_noise(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "use_noise"):
            module.use_noise = False
        if hasattr(module, "noise_std"):
            module.noise_std = 0.0


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
    manifold_torus_coverage: float
    manifold_angle1_coverage: float
    manifold_angle2_coverage: float
    manifold_flow_uniform_norm: float
    manifold_flow_median: float
    manifold_flow_p90: float
    manifold_flow_std: float
    manifold_flow_mean_abs_angle1: float
    manifold_flow_mean_abs_angle2: float
    manifold_reliable: bool


class AutonomousDynamics:
    def __init__(self, model: torch.nn.Module):
        self.model = model
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
        else:
            raise ValueError(f"Unsupported model type: {type(self.base)}")

        self.device = next(self.base.parameters()).device
        self.dtype = next(self.base.parameters()).dtype
        if self.kind == "NeuralODE":
            self.input_size = self.base.ode_func.net[0].in_features - self.base.hidden_size
        else:
            self.input_size = self.base.input_proj.in_features
        self.hidden_size = self.base.hidden_size

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

    def step(self, state: torch.Tensor, input_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_t is None:
            input_t = self.zero_input.repeat(state.shape[0], 1)

        if self.kind == "RDNN":
            return self._rdnn_step(state, input_t)
        if self.kind == "TaskGRU":
            return self._gru_step(state, input_t)
        if self.kind == "NeuralODE":
            return self._neural_ode_step(state, input_t)
        return self._lstm_step(state, input_t)

    def decode_output(self, state: torch.Tensor) -> torch.Tensor:
        if self.kind == "RDNN":
            hidden = state[:, : self.hidden_size]
        elif self.kind == "TaskGRU":
            hidden = state
        elif self.kind == "NeuralODE":
            hidden = state
        else:
            hidden = state[:, : self.hidden_size]
        return self.base.fc_out(hidden)

    def init_state(self, batch_size: int, init_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.init_mapper is not None and init_pos is not None:
            mapped = torch.relu(self.init_mapper(init_pos))
        else:
            mapped = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)

        if self.kind == "RDNN":
            g0 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            return torch.cat([mapped, g0], dim=-1)

        if self.kind == "TaskLSTM":
            c0 = torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            return torch.cat([mapped, c0], dim=-1)

        return mapped


def load_trained_model(
    model_name: str,
    activation_name: str,
    hidden_size: int,
    seed: int,
    base_dir: Path,
    device: str = "cpu",
) -> torch.nn.Module:
    ckpt_candidates = []
    if model_name == "NeuralODE":
        ckpt_candidates.extend(
            [
                base_dir / TASK_NAME / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
                base_dir / TASK_NAME / model_name / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
                base_dir / model_name / TASK_NAME / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt",
            ]
        )
    else:
        ckpt_candidates.append(base_dir / TASK_NAME / model_name / activation_name / f"hidden_{hidden_size}" / f"seed_{seed}" / "checkpoint.pt")

    ckpt_path = next((path for path in ckpt_candidates if path.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found. Tried: {', '.join(str(path) for path in ckpt_candidates)}")

    if model_name == "NeuralODE":
        model = build_neural_ode_model(TASK_NAME, hidden_size, activation_name, use_noise=False, noise_std=0.0)
    else:
        model = build_double_av_model(model_name, hidden_size, activation_name, use_noise=False, noise_std=0.0)
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
    num_samples: int = PAPER_NUM_SAMPLES,
    seq_len: int = 256,
    dt: float = 0.1,
) -> torch.Tensor:
    with torch.no_grad():
        dataset = DoubleAngularVelocityDataset(num_samples=num_samples, seq_len=seq_len, dt=dt)
        velocities = dataset.velocities.to(dyn.device, dtype=dyn.dtype)
        init_pos = dataset.init_pos.to(dyn.device, dtype=dyn.dtype)

        state = dyn.init_state(num_samples, init_pos=init_pos)
        for t in range(velocities.shape[1]):
            state = dyn.step(state, velocities[:, t, :])
        return state


def simulate_autonomous(dyn: AutonomousDynamics, init_states: torch.Tensor, steps: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    states = [init_states]
    outputs = [dyn.decode_output(init_states)]

    state = init_states
    with torch.no_grad():
        for _ in range(steps):
            state = dyn.step(state)
            states.append(state)
            outputs.append(dyn.decode_output(state))

    return torch.stack(states, dim=1), torch.stack(outputs, dim=1)


def _angles_from_outputs(outputs_np: np.ndarray) -> np.ndarray:
    angle1 = np.arctan2(outputs_np[..., 0], outputs_np[..., 1])
    angle2 = np.arctan2(outputs_np[..., 2], outputs_np[..., 3])
    return np.stack([angle1, angle2], axis=-1)


def identify_torus_manifold(
    states: torch.Tensor,
    outputs: torch.Tensor,
    bins: int = PAPER_MANIFOLD_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    states_np = states.detach().cpu().numpy()
    outputs_np = outputs.detach().cpu().numpy()

    angles_np = _angles_from_outputs(outputs_np)
    diff = outputs_np[:, 1:, :] - outputs_np[:, :-1, :]
    speed = np.linalg.norm(diff, axis=-1)

    selected_states: List[np.ndarray] = []
    selected_angles: List[np.ndarray] = []
    selected_speed: List[np.ndarray] = []

    start_t = states_np.shape[1] // 4
    for i in range(states_np.shape[0]):
        traj_speed = speed[i]
        if traj_speed[start_t:].size == 0:
            continue

        mask = np.zeros_like(traj_speed, dtype=bool)
        max_speed = float(np.max(traj_speed[start_t:]))
        for ratio in [PAPER_SLOW_RATIO, 3e-3, 1e-2]:
            thr = max(1e-12, ratio * max_speed)
            mask[:] = False
            mask[start_t:] = traj_speed[start_t:] <= thr
            if mask.sum() >= 8:
                break

        if mask.sum() < 8:
            q = float(np.quantile(traj_speed[start_t:], 0.2))
            mask[:] = False
            mask[start_t:] = traj_speed[start_t:] <= q

        idx = np.where(mask)[0] + 1
        if idx.size == 0:
            continue

        selected_states.append(states_np[i, idx, :])
        selected_angles.append(angles_np[i, idx, :])
        selected_speed.append(traj_speed[idx - 1])

    if not selected_states:
        raise RuntimeError("No slow torus points found; try longer autonomous simulation.")

    selected_states_np = np.concatenate(selected_states, axis=0)
    selected_angles_np = np.concatenate(selected_angles, axis=0)
    selected_speed_np = np.concatenate(selected_speed, axis=0)

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    bin1 = np.clip(np.digitize(selected_angles_np[:, 0], edges, right=False) - 1, 0, bins - 1)
    bin2 = np.clip(np.digitize(selected_angles_np[:, 1], edges, right=False) - 1, 0, bins - 1)

    best_by_bin: Dict[Tuple[int, int], int] = {}
    for idx, key in enumerate(zip(bin1, bin2)):
        prev = best_by_bin.get(key)
        if prev is None or selected_speed_np[idx] < selected_speed_np[prev]:
            best_by_bin[key] = idx

    manifold_indices = sorted(best_by_bin.values(), key=lambda j: (selected_angles_np[j, 0], selected_angles_np[j, 1]))
    manifold_states = selected_states_np[manifold_indices]
    manifold_angles = selected_angles_np[manifold_indices]

    occupied_bins = len(best_by_bin)
    angle1_coverage = len(np.unique(bin1[manifold_indices])) / max(1, bins)
    angle2_coverage = len(np.unique(bin2[manifold_indices])) / max(1, bins)
    torus_coverage = occupied_bins / max(1, bins * bins)

    stats = {
        "occupied_bins": float(occupied_bins),
        "torus_coverage": float(torus_coverage),
        "angle1_coverage": float(angle1_coverage),
        "angle2_coverage": float(angle2_coverage),
    }
    manifold_bin1 = bin1[manifold_indices]
    manifold_bin2 = bin2[manifold_indices]
    return manifold_states, manifold_angles, selected_speed_np[manifold_indices], manifold_bin1, manifold_bin2, stats


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


def detect_fixed_points_on_torus(
    dyn: AutonomousDynamics,
    manifold_states: np.ndarray,
    manifold_angles: np.ndarray,
    flow: np.ndarray,
    flow_norms: np.ndarray,
    bin1: np.ndarray,
    bin2: np.ndarray,
    bins: int,
    threshold_ratio: float = 0.15,
) -> List[Dict[str, object]]:
    if flow_norms.size == 0:
        return []

    finite_norms = flow_norms[np.isfinite(flow_norms)]
    if finite_norms.size == 0:
        return []

    max_norm = float(np.max(finite_norms))
    thr = max(1e-12, max(PAPER_SLOW_RATIO * max_norm, float(np.quantile(finite_norms, threshold_ratio))))

    cell_to_idx: Dict[Tuple[int, int], int] = {}
    for idx, key in enumerate(zip(bin1.tolist(), bin2.tolist())):
        cell_to_idx[key] = idx

    candidates: List[int] = []
    for idx, (i, j) in enumerate(zip(bin1.tolist(), bin2.tolist())):
        if not np.isfinite(flow_norms[idx]) or flow_norms[idx] > thr:
            continue

        local_min = True
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbor = ((i + di) % bins, (j + dj) % bins)
                neighbor_idx = cell_to_idx.get(neighbor)
                if neighbor_idx is None:
                    continue
                if flow_norms[neighbor_idx] < flow_norms[idx] - 1e-12:
                    local_min = False
                    break
            if not local_min:
                break

        if local_min:
            candidates.append(idx)

    if not candidates:
        best = int(np.argmin(flow_norms))
        candidates = [best]

    fixed_points: List[Dict[str, object]] = []
    for idx in candidates:
        state_guess = torch.tensor(manifold_states[idx], device=dyn.device, dtype=dyn.dtype)
        J = jacobian_at_point(dyn, state_guess)
        jac_type, eigvals = classify_fixed_point_from_jacobian(J)
        fixed_points.append(
            {
                "theta1": float(manifold_angles[idx, 0]),
                "theta2": float(manifold_angles[idx, 1]),
                "bin1": int(bin1[idx]),
                "bin2": int(bin2[idx]),
                "flow_norm": float(flow_norms[idx]),
                "flow_vector": flow[idx].tolist(),
                "jacobian_type": jac_type,
                "max_abs_eig": float(np.max(np.abs(eigvals))),
                "jacobian_spectrum": serialize_complex_spectrum(eigvals),
                "damping_ratios": damping_ratios_from_discrete_eigs(eigvals, dt=1.0),
            }
        )

    unique_fixed_points: List[Dict[str, object]] = []
    for fp in fixed_points:
        keep = True
        for other in unique_fixed_points:
            d1 = wrap_angle(np.array([fp["theta1"] - other["theta1"]]))[0]
            d2 = wrap_angle(np.array([fp["theta2"] - other["theta2"]]))[0]
            if math.sqrt(float(d1 * d1 + d2 * d2)) <= 0.25:
                keep = False
                break
        if keep:
            unique_fixed_points.append(fp)

    return unique_fixed_points


def detect_limit_cycles(
    dyn: AutonomousDynamics,
    states: torch.Tensor,
    max_period: Optional[int] = None,
    tol: float = 5e-3,
    min_period: int = 2,
    min_repeats: int = 3,
    min_amplitude_ratio: float = 1e-3,
) -> List[Dict[str, object]]:
    states_np = states.detach().cpu().numpy()
    cycles = []

    for i in range(states_np.shape[0]):
        traj = states_np[i]
        T = traj.shape[0]
        transient = T // 2
        tail = traj[transient:]

        auto_max = tail.shape[0] // (min_repeats + 1)
        p_max = min(auto_max, max_period) if max_period is not None else auto_max
        if p_max < min_period:
            continue

        state_scale = np.mean(np.linalg.norm(tail, axis=1)) + 1e-12
        traj_candidates = []
        for p in range(min_period, p_max + 1):
            if tail.shape[0] < (min_repeats + 1) * p:
                continue

            chunks = [tail[-(k + 1) * p : -k * p if k > 0 else None] for k in range(min_repeats, -1, -1)]
            pair_diffs = []
            for k in range(len(chunks) - 1):
                pair_diffs.append(np.linalg.norm(chunks[k] - chunks[k + 1], axis=1).mean())

            d = float(np.mean(pair_diffs))
            rel_d = d / state_scale
            if d < tol or rel_d < tol:
                cycle_pts = tail[-p:]
                amplitude = float(np.mean(np.linalg.norm(cycle_pts - cycle_pts.mean(axis=0, keepdims=True), axis=1)))
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
            traj_candidates.sort(key=lambda c: (c["period"], -c["amplitude"], -c["rel_error"]))
            cycles.append(traj_candidates[-1])

    unique_cycles = []
    signatures = []
    for c in cycles:
        pts = c["points"]
        sig = np.mean(np.linalg.norm(pts - pts.mean(axis=0, keepdims=True), axis=1))
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


def compute_torus_flow(dyn: AutonomousDynamics, manifold_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    state_tensor = torch.tensor(manifold_states, device=dyn.device, dtype=dyn.dtype)
    with torch.no_grad():
        next_state = dyn.step(state_tensor)
        outputs = dyn.decode_output(state_tensor).detach().cpu().numpy()
        next_outputs = dyn.decode_output(next_state).detach().cpu().numpy()

    angles = _angles_from_outputs(outputs)
    next_angles = _angles_from_outputs(next_outputs)
    dtheta = wrap_angle(next_angles - angles)
    norms = np.linalg.norm(dtheta, axis=1)
    return dtheta, norms


def analyze_checkpoint(
    model_name: str,
    activation_name: str,
    hidden_size: int,
    seed: int,
    base_dir: str = "contrast_experiment_weights",
    device: str = "cpu",
    num_samples: int = PAPER_NUM_SAMPLES,
    seq_len: int = 256,
    dt: float = 0.1,
    autonomous_steps: Optional[int] = None,
    manifold_bins: int = PAPER_MANIFOLD_BINS,
) -> Dict[str, object]:
    base_path = Path(base_dir)
    model = load_trained_model(model_name, activation_name, hidden_size, seed, base_path, device=device)
    dyn = AutonomousDynamics(model)

    if autonomous_steps is None:
        autonomous_steps = seq_len * 32

    end_states = collect_task_end_states(dyn, num_samples=num_samples, seq_len=seq_len, dt=dt)
    states, outputs = simulate_autonomous(dyn, end_states, steps=autonomous_steps)

    manifold_states, manifold_angles, selected_speed, manifold_bin1, manifold_bin2, manifold_stats = identify_torus_manifold(
        states,
        outputs,
        bins=manifold_bins,
    )
    dtheta, flow_norms = compute_torus_flow(dyn, manifold_states)

    fixed_points = detect_fixed_points_on_torus(
        dyn=dyn,
        manifold_states=manifold_states,
        manifold_angles=manifold_angles,
        flow=dtheta,
        flow_norms=flow_norms,
        bin1=manifold_bin1,
        bin2=manifold_bin2,
        bins=manifold_bins,
    )
    cycles = detect_limit_cycles(dyn, states, max_period=min(512, max(2, autonomous_steps // 2)))

    count_stable = sum(1 for f in fixed_points if f["jacobian_type"] == "stable")
    count_saddle = sum(1 for f in fixed_points if f["jacobian_type"] == "saddle")
    count_unstable = sum(1 for f in fixed_points if f["jacobian_type"] == "unstable")
    count_marginal = sum(1 for f in fixed_points if f["jacobian_type"] == "marginal")

    count_cycle_stable = sum(1 for c in cycles if c["stability"] == "stable")
    count_cycle_uncertain = sum(1 for c in cycles if c["stability"] == "uncertain")

    summary = DynamicsResult(
        task=TASK_NAME,
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
        manifold_points=int(len(manifold_states)),
        manifold_torus_coverage=float(manifold_stats["torus_coverage"]),
        manifold_angle1_coverage=float(manifold_stats["angle1_coverage"]),
        manifold_angle2_coverage=float(manifold_stats["angle2_coverage"]),
        manifold_flow_uniform_norm=float(np.max(flow_norms)),
        manifold_flow_median=float(np.median(flow_norms)),
        manifold_flow_p90=float(np.percentile(flow_norms, 90)),
        manifold_flow_std=float(np.std(flow_norms)),
        manifold_flow_mean_abs_angle1=float(np.mean(np.abs(dtheta[:, 0]))),
        manifold_flow_mean_abs_angle2=float(np.mean(np.abs(dtheta[:, 1]))),
        manifold_reliable=bool(len(manifold_states) >= max(24, manifold_bins) and manifold_stats["torus_coverage"] >= 0.05),
    )

    return {
        "summary": summary.__dict__,
        "fixed_points": fixed_points,
        "limit_cycles": cycles,
        "manifold_states": manifold_states.tolist(),
        "manifold_angles": manifold_angles.tolist(),
        "manifold_speed": selected_speed.tolist(),
        "manifold_bins": {"bin1": manifold_bin1.tolist(), "bin2": manifold_bin2.tolist()},
        "flow": dtheta.tolist(),
        "flow_norms": flow_norms.tolist(),
        "manifold_stats": manifold_stats,
    }
