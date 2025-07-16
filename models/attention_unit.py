import torch
import torch.nn as nn
import torch.nn.functional as F


class SGUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SGUnit, self).__init__()
        self.projection_1 = nn.Linear(input_dim, input_dim)
        self.SGLayer = nn.Linear(input_dim, input_dim)
        self.projection_2 = nn.Linear(input_dim, input_dim)
        self.embed = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        z = self.projection_1(x)
        z = F.gelu(z)
        s = self.SGLayer(z)
        y = self.projection_2(s)
        y = y + x
        out = self.embed(y)
        out = F.gelu(out)

        return out