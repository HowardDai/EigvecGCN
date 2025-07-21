import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_unit import SGUnit




###MLP with lienar output
class AttentionMLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, evec_len):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(AttentionMLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers
        self.initial_attention = SGUnit(input_dim, hidden_dim, evec_len)

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, evec_len)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.layers = torch.nn.ModuleList()
        
            for layer in range(num_layers - 2):
                self.layers.append(SGUnit(hidden_dim, hidden_dim, evec_len))
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, evec_len))

    def forward(self, x, edge_index=None, batch=None):


        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            h = self.initial_attention(h)

            for layer in range(0, 2* (self.num_layers - 2), 2):
                h = (self.layers[layer](h))
                h = F.relu(self.layers[layer+1](h))
                
            return self.layers[-1](h)