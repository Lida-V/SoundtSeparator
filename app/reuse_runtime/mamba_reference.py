"""Mamba-1 の推論用 PyTorch 実装。カスタム CUDA / Triton を必要としない。

重みのキーと演算順序は mamba-ssm の Mamba モジュールに合わせている。
高速な専用カーネルではないため、長い音声は分割処理する。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _selective_scan(x: torch.Tensor, delta: torch.Tensor, a: torch.Tensor,
                    b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    # x,delta: [batch, channels, time]; b,c: [batch,state,time]
    state = torch.zeros((x.shape[0], x.shape[1], a.shape[1]), device=x.device, dtype=x.dtype)
    output = torch.jit.annotate(list[torch.Tensor], [])
    for t in range(x.shape[2]):
        dt = delta[:, :, t]
        state = (torch.exp(dt.unsqueeze(-1) * a.unsqueeze(0)) * state
                 + dt.unsqueeze(-1) * b[:, :, t].unsqueeze(1) * x[:, :, t].unsqueeze(-1))
        output.append((state * c[:, :, t].unsqueeze(1)).sum(-1) + d * x[:, :, t])
    return torch.stack(output, dim=2)


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):
        length = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :length])
        delta, b, c = torch.split(self.x_proj(x.transpose(1, 2)),
                                  [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta)).transpose(1, 2)
        y = _selective_scan(x, delta, -torch.exp(self.A_log.float()),
                            b.transpose(1, 2), c.transpose(1, 2), self.D.float())
        return self.out_proj((y * F.silu(z)).transpose(1, 2))
