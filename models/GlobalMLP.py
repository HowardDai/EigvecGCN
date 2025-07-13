import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalMLP(nn.module):
    def __init__(self):
        super(GlobalMLP, self).__init__()

        #LINEAR LAYERS
        self.linear1 = nn.Linear(252, 400)
        self.linear2 = nn.Linear(400, 600)
        self.linear3 = nn.Linear(600, 800)
        self.linear4 = nn.Linear(8000, 1200)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.Relu(x)
        x = self.linear2(x)
        x = nn.ReLu(x)
        x = self.linear3(x)
        x = nn.Relu(x)
        x = self.linear4(x)
        
        return x

