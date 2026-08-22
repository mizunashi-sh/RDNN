import math

import torch
from torch.utils.data import Dataset


class AngularVelocityDataset(Dataset):
    def __init__(self, num_samples, seq_len=256, dt=0.1, min_semicircle_fraction=1.2, max_semicircle_fraction=4.0):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dt = dt

        t = torch.linspace(0, seq_len * dt, seq_len)

        base_vel = torch.zeros(num_samples, seq_len)
        for _ in range(3):
            freq = 0.2 + torch.rand(num_samples, 1) * 1.8
            phase = torch.rand(num_samples, 1) * 2 * math.pi
            amp = (0.5 + torch.rand(num_samples, 1) * 1.5) * torch.sign(torch.randn(num_samples, 1))
            base_vel += amp * torch.sin(2 * math.pi * freq * t + phase)

        min_total_rotation = min_semicircle_fraction * math.pi
        max_total_rotation = max_semicircle_fraction * math.pi
        target_abs = min_total_rotation + (max_total_rotation - min_total_rotation) * torch.rand(num_samples)
        target_sign = torch.where(torch.rand(num_samples) > 0.5, 1.0, -1.0)
        target_delta = target_abs * target_sign

        base_delta = base_vel.sum(dim=1) * dt
        bias = (target_delta - base_delta) / (seq_len * dt)
        self.velocities = base_vel + bias.unsqueeze(1)

        self.theta_0 = torch.rand(num_samples) * 2 * math.pi
        self.thetas = self.theta_0.unsqueeze(1) + torch.cumsum(self.velocities * dt, dim=1)

        self.targets = torch.stack([torch.sin(self.thetas), torch.cos(self.thetas)], dim=-1)
        self.velocities = self.velocities.unsqueeze(-1)
        self.init_pos = torch.stack([torch.sin(self.theta_0), torch.cos(self.theta_0)], dim=-1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.velocities[idx], self.init_pos[idx], self.targets[idx], self.thetas[idx]
    

class SaccadeDataset(Dataset):
    def __init__(self, num_samples, seq_len=512):
        self.num_samples = num_samples
        self.seq_len = seq_len

        self.inputs = torch.zeros(num_samples, seq_len, 3)
        self.targets = torch.zeros(num_samples, seq_len, 2)
        self.masks = torch.ones(num_samples, seq_len, 1)
        self.thetas = torch.zeros(num_samples)

        for i in range(num_samples):
            theta = torch.rand(1).item() * 2 * math.pi
            self.thetas[i] = theta

            stim_duration = 15
            cue_duration = 5
            delay_duration = torch.randint(50, 400, (1,)).item()
            go_cue_time = stim_duration + delay_duration

            self.inputs[i, :stim_duration, 0] = math.sin(theta)
            self.inputs[i, :stim_duration, 1] = math.cos(theta)

            self.targets[i, :go_cue_time, 0] = 0.0
            self.targets[i, :go_cue_time, 1] = 0.0

            self.targets[i, go_cue_time:, 0] = math.sin(theta)
            self.targets[i, go_cue_time:, 1] = math.cos(theta)
            self.masks[i, go_cue_time:go_cue_time+cue_duration, 0] = 0.0
            self.inputs[i, go_cue_time:go_cue_time+cue_duration, 2] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.masks[idx], self.thetas[idx]


class SaccadeDataset(Dataset):
    def __init__(self, num_samples, seq_len=512):
        self.num_samples = num_samples
        self.seq_len = seq_len

        self.inputs = torch.zeros(num_samples, seq_len, 3)
        self.targets = torch.zeros(num_samples, seq_len, 2)
        self.masks = torch.ones(num_samples, seq_len, 1)
        self.thetas = torch.zeros(num_samples)

        for i in range(num_samples):
            theta = torch.rand(1).item() * 2 * math.pi
            self.thetas[i] = theta

            stim_duration = 15
            cue_duration = 5
            delay_duration = torch.randint(50, 400, (1,)).item()
            go_cue_time = stim_duration + delay_duration

            self.inputs[i, :stim_duration, 0] = math.sin(theta)
            self.inputs[i, :stim_duration, 1] = math.cos(theta)

            self.targets[i, go_cue_time:, 0] = math.sin(theta)
            self.targets[i, go_cue_time:, 1] = math.cos(theta)
            self.masks[i, go_cue_time:go_cue_time+cue_duration, 0] = 0.0
            self.inputs[i, go_cue_time:go_cue_time+cue_duration, 2] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.masks[idx], self.thetas[idx]
    
class DoubleAngularVelocityDataset(Dataset):
    def __init__(self, num_samples, seq_len=256, dt=0.1, min_semicircle_fraction=1.2, max_semicircle_fraction=4.0):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dt = dt

        t = torch.linspace(0, seq_len * dt, seq_len)

        # 辅助函数：生成单条平滑的角速度曲线
        def generate_single_velocity():
            base_vel = torch.zeros(num_samples, seq_len)
            for _ in range(3):
                freq = 0.2 + torch.rand(num_samples, 1) * 1.8
                phase = torch.rand(num_samples, 1) * 2 * math.pi
                amp = (0.5 + torch.rand(num_samples, 1) * 1.5) * torch.sign(torch.randn(num_samples, 1))
                base_vel += amp * torch.sin(2 * math.pi * freq * t + phase)

            # 约束角位移
            min_total_rotation = min_semicircle_fraction * math.pi
            max_total_rotation = max_semicircle_fraction * math.pi
            target_abs = min_total_rotation + (max_total_rotation - min_total_rotation) * torch.rand(num_samples)
            target_sign = torch.where(torch.rand(num_samples) > 0.5, 1.0, -1.0)
            target_delta = target_abs * target_sign

            base_delta = base_vel.sum(dim=1) * dt
            bias = (target_delta - base_delta) / (seq_len * dt)
            return base_vel + bias.unsqueeze(1)

        # 独立生成两组速度
        vel1 = generate_single_velocity()
        vel2 = generate_single_velocity()

        self.velocities = torch.stack([vel1, vel2], dim=-1)

        self.theta1_0 = torch.rand(num_samples) * 2 * math.pi
        self.theta2_0 = torch.rand(num_samples) * 2 * math.pi

        self.init_pos = torch.stack([
            torch.sin(self.theta1_0), torch.cos(self.theta1_0),
            torch.sin(self.theta2_0), torch.cos(self.theta2_0)
        ], dim=-1)

        self.theta1s = self.theta1_0.unsqueeze(1) + torch.cumsum(vel1 * self.dt, dim=1)
        self.theta2s = self.theta2_0.unsqueeze(1) + torch.cumsum(vel2 * self.dt, dim=1)
        self.thetas = torch.stack([self.theta1s, self.theta2s], dim=-1)

        self.targets = torch.stack([
            torch.sin(self.theta1s), torch.cos(self.theta1s),
            torch.sin(self.theta2s), torch.cos(self.theta2s)
        ], dim=-1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.velocities[idx], self.init_pos[idx], self.targets[idx], self.thetas[idx]
