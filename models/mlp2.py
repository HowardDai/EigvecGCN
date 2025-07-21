import torch
import torch.nn as nn
import torch.nn.functional as F

###MLP2 with lienar output
class MLP2(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.3):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
            dropout: dropout rate to apply after each hidden layer
        '''
    
        super(MLP2, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers
        self.dropout = dropout

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.layer_norms = torch.nn.ModuleList()
            self.dropouts = torch.nn.ModuleList()
            self.skip_projections = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))

            if input_dim != hidden_dim:
                self.skip_projections.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.skip_projections.append(nn.Identity())

            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.skip_projections.append(nn.Identity())
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.layer_norms.append(nn.LayerNorm((hidden_dim)))
                self.dropouts.append(nn.Dropout(p=dropout))

    def forward(self, x, edge_index=None, batch=None):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h_res = self.skip_projections[layer](h)
                h = self.layer_norms[layer](self.linears[layer](h))
                h = F.relu(h)
                h = h + h_res
                h = F.relu(h)
                h = self.dropouts[layer](h)
            return self.linears[self.num_layers - 1](h)