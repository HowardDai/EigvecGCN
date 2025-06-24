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

    model = GIN(8, 3, 1, 60, config.num_eigenvectors, 0.1, True, "Sum", device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    
                           
    os.makedirs("checkpoints", exist_ok=True)
    
    training_loop(model, train_loader, val_loader, optimizer, device, config)


    # if not config.multiple_runs:
    #     print("Started training with 1 run.",
    #           f"Early stopping: {'Yes' if config.use_early_stopping else 'No'}")
    #     val_acc, val_loss = training_loop(model, features, labels, adj, train_set_ind, val_set_ind, config)
    #     out_features = evaluate_on_test(model, features, labels, adj, test_set_ind, config)

    #     visualize_validation_performance(val_acc, val_loss)
    #     visualize_embedding_tSNE(labels, out_features, NUM_CLASSES)
    # else:
    #     print(f"Started training with {config.num_of_runs} runs.",
    #           f"Early stopping: {'Yes' if config.use_early_stopping else 'No'}")
    #     multiple_runs(model, features, labels, adj,
    #                   [train_set_ind, val_set_ind, test_set_ind],
    #                   config, training_loop, evaluate_on_test)
