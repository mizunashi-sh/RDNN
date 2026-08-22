import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

class ODEFunc(nn.Module):
    def __init__(self, hidden_size, input_size, act_fn):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.Tanh() if act_fn == 'tanh' else nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.x_t = None

    def forward(self, t, h):
        if self.x_t is None:
            raise RuntimeError("Please ensure that the current step input x_t is set before calling the ODE solver.")
            
        return self.net(torch.cat([h, self.x_t], dim=-1))


class NeuralODE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 method='rk4', use_adjoint=False, use_noise=False, noise_std=0.0, act_fn='relu'):
        super(NeuralODE, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        self.use_adjoint = use_adjoint
        self.use_noise = use_noise
        self.noise_std = noise_std
        
        self.ode_func = ODEFunc(hidden_size, input_size, act_fn)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.register_buffer("integration_time", torch.tensor([0.0, 0.01]))

    def forward(self, x, R_0=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        solver = odeint_adjoint if self.use_adjoint else odeint
        
        hidden = R_0 if R_0 is not None else torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        
        outputs = []
        for input_t in x.unbind(dim=1):
            self.ode_func.x_t = input_t

            out = solver(self.ode_func, hidden, self.integration_time, method=self.method)
            hidden = out[-1]  # 形状为 [batch_size, hidden_size]
            
            if self.use_noise:
                hidden = hidden + self.noise_std * torch.randn_like(hidden)
                
            outputs.append(hidden)
            
        self.ode_func.x_t = None
        
        outputs_tensor = torch.stack(outputs, dim=1)
        logits_seq = self.fc_out(outputs_tensor)
        return logits_seq, outputs_tensor
    