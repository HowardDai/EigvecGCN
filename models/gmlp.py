
import torch
import torch.nn as nn


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()

        self.norm = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v


class GatingMlpBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, survival_prob):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)
        self.prob = survival_prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
            return x
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut


class gMLP(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        seq_len,
        n_blocks,
        output_dim,
        prob_0_L=[1, 0.5],
    ):
        super().__init__()

        self.survival_probs = torch.linspace(prob_0_L[0], prob_0_L[1], n_blocks)
        self.blocks = nn.ModuleList(
            [GatingMlpBlock(d_model, d_ffn, seq_len, prob) for prob in self.survival_probs]
        )
        self.linear = nn.Linear(d_model, 2*output_dim)
        self.linear2 = nn.Linear(2*output_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        for gmlp_block in self.blocks:
            x = gmlp_block(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
