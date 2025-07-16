import torch
import torch.nn as nn
import torch.nn.functional as F


class SGUnit(nn.Module):
    def __init__(self, input_dim, output_dim, evec_len):
        super(SGUnit, self).__init__()
        self.U = nn.Linear(input_dim, evec_len)
        self.V = nn.Lineat(evec_len, input_dim)
        self.SGLayer = nn.Linear(input_dim, input_dim)
        torch.nn.init.zeros_(self.SGLayer.weight)
        torch.nn.init.ones_(self.SGLayer.bias)
        self.projection_2 = nn.Linear(input_dim, input_dim)
        self.embed = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        z = self.U(x)
        z = F.gelu(z)
        s = self.SGLayer.weights @ z + self.SGLayer.bias
        s = s * z
        y = self.V(s)
        y = y + x
        out = self.embed(y)
        out = F.gelu(out)

        return out