import sys
sys.path.append("models/")
from mlp import MLP
from mlp2 import MLP2
from GCN import GCN
from GIN import GIN
from harmonic import HarmonicAlgorithm

from utils import *
from globals import config
from training_evaluation import *
from visualization import *

import os



if __name__ == "__main__":
    
    if config.load_model != None:
        assert(os.path.exists(config.load_model))
    
    # LOADING DATASET
    
    data_dict_emb, train_loader, val_loader, test_loader = load_data(config)

    device = "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert(len(data_dict_emb.keys()) > 0)

    sample_set = data_dict_emb[next(iter(data_dict_emb))] # get one of train, valid, test, depending on what exists in the data_dict_emb
    sample = sample_set[0] 
    input_size = sample.x.shape[-1]
    print("input_size", input_size)
    

    if config.model == 'GCN':
        model = GCN(input_size, config.num_eigenvectors, config.dropout, config.use_bias).to(device)
    elif config.model == "GIN":
        model = GIN(8, 3, input_size, 60, config.num_eigenvectors, 0.1, True, "Sum", device).to(device) # TODO: make these hyperparams configurable in command line 
    elif config.model == "MLP":
        model = MLP(8, input_size, 60, config.num_eigenvectors).to(device)
    elif config.model == "MLP2":
        model = MLP2(8, input_size, 60, config.num_eigenvectors, config.dropout).to(device)
    elif config.model == "harmonic":
        model = HarmonicAlgorithm(config.num_eigenvectors)
    else:
        print("Invalid model type")
    

    
    if config.load_model != None: # TODO: make this flexible, maybe it checks for checkpoint_folder/load_model if just loading the path given by load_model doesn't work 
        print(f"Loading checkpoint: {config.load_model}")
        model.load_state_dict(torch.load(config.load_model, weights_only=True))

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(f"checkpoints/{config.checkpoint_folder}", exist_ok=True)

    if config.train: # only need if training. This throws error if trying to use HarmonicAlgorithm for training 
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)




    os.makedirs('plots', exist_ok=True)
    os.makedirs(f"plots/{config.checkpoint_folder}", exist_ok=True)


    if config.train:
        print("training...")
        training_loop(model, train_loader, val_loader, optimizer, device, config)

    if config.test:
        print("testing...")
        evaluate(model, test_loader, device, config)



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
