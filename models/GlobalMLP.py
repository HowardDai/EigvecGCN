import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalMLP(nn.Module):
    def __init__(self):
        super(GlobalMLP, self).__init__()

        #LINEAR LAYERS
        self.linear1 = nn.Linear(253, 400)
        self.linear2 = nn.Linear(400, 600)
        self.linear3 = nn.Linear(600, 600)
        self.linear4 = nn.Linear(600, 240)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)

        x = torch.reshape(x, (x.shape[0], 40, 6))
        
        return x

