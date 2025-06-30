import sys
sys.path.append("models/")
from mlp import MLP
from GCN import GCN
from GIN import GIN

from utils import *
from globals import config
from training_evaluation import *
from visualization import *

import os



if __name__ == "__main__":
   

    dataset, train_loader, val_loader, test_loader = load_data(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = GCN(1, config.num_eigenvectors, config.dropout, config.use_bias)

    sample = dataset[0] 
    input_size = sample.x.shape[-1]
    print("input_size", input_size)

    if config.model == 'GCN':
        model = GCN(input_size, config.num_eigenvectors, config.dropout, config.use_bias)
    elif config.model == "GIN":
        model = GIN(8, 3, input_size, 60, config.num_eigenvectors, 0.1, True, "Sum", device)
    elif config.model == "MLP":
        model = MLP(8, input_size, 60, config.num_eigenvectors)
    else:
        print("Invalid model type")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    
    
    os.makedirs(config.checkpoint_folder, exist_ok=True)
    print("Started training with 1 run.",
               f"Early stopping: {'Yes' if config.use_early_stopping else 'No'}")

    training_loop(model, train_loader, val_loader, optimizer, device, config)