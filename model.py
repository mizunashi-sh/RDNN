import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dale_principle=True, subtractive=False, static_gating=False, use_noise=False, noise_std=0., act_fn=torch.tanh):
        super(RDNN, self).__init__()
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.noise_std = noise_std
        self.use_noise = use_noise
        self.static_gating = static_gating
        self.subtractive = subtractive
        self.dale_principle = dale_principle

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.raw_J = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.raw_w = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.raw_eta = nn.Parameter(torch.empty(hidden_size))
        self.raw_alpha_r = nn.Parameter(torch.empty(hidden_size)) 
        self.raw_alpha_g = nn.Parameter(torch.empty(hidden_size))
        
        self._reset_parameters()

    def _reset_parameters(self):
        init_mean = -math.log(self.hidden_size)
        init_std = 1.0 / math.sqrt(self.hidden_size)
        
        nn.init.normal_(self.raw_J, mean=init_mean, std=init_std)
        nn.init.normal_(self.raw_w, mean=init_mean, std=init_std)

        nn.init.normal_(self.raw_alpha_r, mean=-2.2, std=0.1)
        nn.init.normal_(self.raw_alpha_g, mean=-2.2, std=0.1)

        nn.init.normal_(self.raw_eta, mean=0.0, std=0.1)

    def forward(self, x, R_0=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        I_seq = self.input_proj(x)  # [B, T, H]

        J_pos = F.softplus(self.raw_J) if self.dale_principle else self.raw_J
        w_pos = F.softplus(self.raw_w) if self.dale_principle else self.raw_w
        eta = F.softplus(self.raw_eta) + 1e-5 if self.dale_principle else self.raw_eta + 1e-5
        alpha_r = torch.sigmoid(self.raw_alpha_r)
        alpha_g = torch.sigmoid(self.raw_alpha_g)
        
        R_t = R_0 if R_0 is not None else torch.zeros(batch_size, self.hidden_size, device=device)
        G_t = torch.zeros(batch_size, self.hidden_size, device=device)

        I_seq_unbound = I_seq.unbind(dim=1) 

        outputs =[]
        
        for I_t in I_seq_unbound:
            firing_rate = self.act_fn(R_t)
            
            recurrent_drive = F.linear(firing_rate, J_pos)
            
            if self.static_gating:
                target_R = (recurrent_drive + I_t) / eta
            elif self.subtractive:
                target_R = recurrent_drive + I_t - G_t
            else:
                target_R = (recurrent_drive + I_t) / (eta + G_t)
            if self.use_noise:
                target_R = target_R + self.noise_std * torch.randn_like(R_t)

            R_t = (1.0 - alpha_r) * R_t + alpha_r * target_R
            if self.dale_principle:
                R_t = F.relu(R_t)

            target_G = F.linear(firing_rate, w_pos)
            if self.use_noise:
                target_G = target_G + self.noise_std * torch.randn_like(G_t)

            G_t = (1.0 - alpha_g) * G_t + alpha_g * target_G
            if self.dale_principle:
                G_t = F.relu(G_t)
            
            outputs.append(R_t)
            
        outputs_tensor = torch.stack(outputs, dim=1)
        
        logits_seq = self.fc_out(outputs_tensor) 
        
        return logits_seq, outputs_tensor
    

class LowRankRDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rank=10, use_noise=False, noise_std=0., act_fn=torch.tanh):
        super(LowRankRDNN, self).__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.act_fn = act_fn
        self.noise_std = noise_std
        self.use_noise = use_noise

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.raw_J1 = nn.Parameter(torch.empty(hidden_size, rank))
        self.raw_J2 = nn.Parameter(torch.empty(rank, hidden_size))
        
        self.raw_w1 = nn.Parameter(torch.empty(hidden_size, rank))
        self.raw_w2 = nn.Parameter(torch.empty(rank, hidden_size))
        
        self.raw_eta = nn.Parameter(torch.empty(hidden_size))
        self.raw_alpha_r = nn.Parameter(torch.empty(hidden_size)) 
        self.raw_alpha_g = nn.Parameter(torch.empty(hidden_size))
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.raw_J1, mean=-1.0, std=1.0 / math.sqrt(self.rank))
        nn.init.normal_(self.raw_J2, mean=-1.0, std=1.0 / math.sqrt(self.hidden_size))

        nn.init.normal_(self.raw_w1, mean=-1.0, std=1.0 / math.sqrt(self.rank))
        nn.init.normal_(self.raw_w2, mean=-1.0, std=1.0 / math.sqrt(self.hidden_size))
        
        nn.init.normal_(self.raw_alpha_r, mean=-2.2, std=0.1)
        nn.init.normal_(self.raw_alpha_g, mean=-2.2, std=0.1)
        
        nn.init.normal_(self.raw_eta, mean=0.0, std=0.1)

    def forward(self, x, R_0=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        I_seq = self.input_proj(x)  # [B, T, H]

        J1_pos = F.softplus(self.raw_J1)
        J2_pos = F.softplus(self.raw_J2)
        w1_pos = F.softplus(self.raw_w1)
        w2_pos = F.softplus(self.raw_w2)
        
        eta = F.softplus(self.raw_eta) + 1e-5
        alpha_r = torch.sigmoid(self.raw_alpha_r)
        alpha_g = torch.sigmoid(self.raw_alpha_g)
        
        R_t = R_0 if R_0 is not None else torch.zeros(batch_size, self.hidden_size, device=device)
        G_t = torch.zeros(batch_size, self.hidden_size, device=device)

        I_seq_unbound = I_seq.unbind(dim=1) 

        outputs = []
        
        for I_t in I_seq_unbound:
            firing_rate = self.act_fn(R_t)
            
            recurrent_drive = F.linear(firing_rate, J1_pos @ J2_pos)
            
            target_R = (recurrent_drive + I_t) / (eta + G_t)
            if self.use_noise:
                target_R = target_R + self.noise_std * torch.randn_like(R_t)

            R_t = F.relu((1.0 - alpha_r) * R_t + alpha_r * target_R)

            target_G = F.linear(firing_rate, w1_pos @ w2_pos)
            if self.use_noise:
                target_G = target_G + self.noise_std * torch.randn_like(G_t)
            G_t = F.relu((1.0 - alpha_g) * G_t + alpha_g * target_G)
            
            outputs.append(R_t)
            
        outputs_tensor = torch.stack(outputs, dim=1)
        
        logits_seq = self.fc_out(outputs_tensor) 
        
        return logits_seq, outputs_tensor


class TaskRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_noise=False, noise_std=0., act_fn=torch.tanh):
        super(TaskRNN, self).__init__()
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.noise_std = noise_std
        self.use_noise = use_noise

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size, output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.hidden_proj.weight, std=1.5 / math.sqrt(self.hidden_size))

        with torch.no_grad():
            self.input_proj.bias[self.hidden_size:2 * self.hidden_size].fill_(1.0)
        
    def forward(self, x, R_0=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        hidden = R_0 if R_0 is not None else torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)

        outputs = []
        for input_t in x.unbind(dim=1):
            preact = self.input_proj(input_t) + self.hidden_proj(hidden)
            if self.use_noise:
                preact = preact + self.noise_std * torch.randn_like(preact)

            hidden = self.act_fn(preact)

            outputs.append(hidden)

        outputs_tensor = torch.stack(outputs, dim=1)
        logits_seq = self.fc_out(outputs_tensor)
        return logits_seq, outputs_tensor


class TaskGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_noise=False, noise_std=0., act_fn=torch.tanh):
        super(TaskGRU, self).__init__()
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.noise_std = noise_std
        self.use_noise = use_noise

        self.input_proj = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size, output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.hidden_proj.weight, std=1.5 / math.sqrt(self.hidden_size))

        with torch.no_grad():
            self.input_proj.bias[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x, R_0=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        hidden = R_0 if R_0 is not None else torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)

        outputs = []
        for input_t in x.unbind(dim=1):
            input_gates = self.input_proj(input_t)
            hidden_gates = self.hidden_proj(hidden)

            input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)

            reset_gate = torch.sigmoid(input_reset + hidden_reset)
            update_gate = torch.sigmoid(input_update + hidden_update)

            candidate = self.act_fn(input_new + reset_gate * hidden_new)
            hidden = (1.0 - update_gate) * hidden + update_gate * candidate
            if self.use_noise:
                hidden = hidden + self.noise_std * torch.randn_like(hidden)

            outputs.append(hidden)

        outputs_tensor = torch.stack(outputs, dim=1)
        logits_seq = self.fc_out(outputs_tensor)
        return logits_seq, outputs_tensor


class TaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_noise=False, noise_std=0., act_fn=torch.tanh):
        super(TaskLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.noise_std = noise_std
        self.use_noise = use_noise

        self.input_proj = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size, output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.hidden_proj.weight, std=1.5 / math.sqrt(self.hidden_size))

        with torch.no_grad():
            self.input_proj.bias[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x, R_0=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        if isinstance(R_0, tuple):
            hidden, cell = R_0
        else:
            hidden = R_0 if R_0 is not None else torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
            cell = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)

        outputs = []
        for input_t in x.unbind(dim=1):
            input_gates = self.input_proj(input_t)
            hidden_gates = self.hidden_proj(hidden)

            input_gate, forget_gate, candidate_gate, output_gate = input_gates.chunk(4, dim=-1)
            hidden_input, hidden_forget, hidden_candidate, hidden_output = hidden_gates.chunk(4, dim=-1)

            input_gate = torch.sigmoid(input_gate + hidden_input)
            forget_gate = torch.sigmoid(forget_gate + hidden_forget)
            output_gate = torch.sigmoid(output_gate + hidden_output)

            candidate = self.act_fn(candidate_gate + hidden_candidate)
            candidate = candidate + self.noise_std * torch.randn_like(candidate)
            
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * self.act_fn(cell)
            if self.use_noise:
                cell = cell + self.noise_std * torch.randn_like(cell)
                hidden = hidden + self.noise_std * torch.randn_like(hidden)
    
            outputs.append(hidden)

        outputs_tensor = torch.stack(outputs, dim=1)
        logits_seq = self.fc_out(outputs_tensor)
        return logits_seq, outputs_tensor


class LowRankTaskRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rank=4, use_noise=False, noise_std=0., act_fn=torch.tanh):
        super(LowRankTaskRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.act_fn = act_fn
        self.noise_std = noise_std
        self.use_noise = use_noise

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.hidden_proj1 = nn.Linear(hidden_size, rank, bias=False)
        self.hidden_proj2 = nn.Linear(rank, hidden_size, bias=False)
        self.fc_out = nn.Linear(hidden_size, output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.hidden_proj1.weight, std=1.5 / math.sqrt(self.rank))
        nn.init.normal_(self.hidden_proj2.weight, std=1.5 / math.sqrt(self.hidden_size))

        with torch.no_grad():
            self.input_proj.bias[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x, R_0=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        hidden = R_0 if R_0 is not None else torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)

        outputs = []
        for input_t in x.unbind(dim=1):
            preact = self.input_proj(input_t) + self.hidden_proj2(self.hidden_proj1(hidden))
            if self.use_noise:
                preact = preact + self.noise_std * torch.randn_like(preact)

            hidden = self.act_fn(preact)
            outputs.append(hidden)

        outputs_tensor = torch.stack(outputs, dim=1)
        logits_seq = self.fc_out(outputs_tensor)
        return logits_seq, outputs_tensor
    