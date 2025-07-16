import sys
sys.path.append("models/")
from mlp import MLP
from mlp2 import MLP2
from GCN import GCN
from GIN import GIN
from GIN import GIN2
from GIN import RecurrentGIN 
from GlobalGIN import GlobalGIN
from GlobalMLP import GlobalMLP

from analytic import HarmonicAlgorithm, GroundTruth, LanczosAlgorithm, RandomVectors

from utils import *
from globals import args
from training_evaluation import *
from visualization import *
from data import *
from global_training import *

import os

from easydict import EasyDict
import yaml



if __name__ == "__main__":
    config = {}
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    

    if config.load_model != None:
        assert(os.path.exists(config.load_model))
    
    if config.checkpoint_folder == None:
        config.checkpoint_folder = os.path.basename(args.config)[:-4] # copies filename of yml file, without the .yml extension
    

    if config.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else: 
        device = "cpu"

    print(f"using: {device}")


    # LOADING DATASET

    if config.dataset == "ogbg_ppa":
        config.evec_len = 300
    elif config.dataset == "zinc":
        config.evec_len = 40

    

    if config.model == "GlobalMLP":
        train_loader, val_loader, test_loader = load_data_global(config)
    else:
        data_dict_emb, train_loader, val_loader, test_loader = load_data(config)
        assert(len(data_dict_emb.keys()) > 0)

        sample_set = data_dict_emb[next(iter(data_dict_emb))] # get one of train, valid, test, depending on what exists in the data_dict_emb
        sample = sample_set[0] 
        input_size = sample.x.shape[-1]
        print("input_size", input_size)

    
    
    # if config.model == 'GCN':
        
    #     model = GCN(input_size, config.num_eigenvectors, config.dropout, config.use_bias).to(device)

    if config.model == "GlobalMLP":
        model = GlobalMLP().to(device)
    if config.model == "GIN":
        model_config = config.GIN_params
        model = GIN(model_config.num_layers, model_config.num_mlp_layers, input_size, model_config.hidden_dim, config.num_eigenvectors, model_config.dropout, True, "Sum", device).to(device) 
    elif config.model == "GIN2":
        model_config = config.GIN2_params
        model = GIN2(model_config.num_layers, model_config.num_mlp_layers, input_size, model_config.hidden_dim, config.num_eigenvectors, model_config.global_dim, model_config.dropout, True, "Sum", device).to(device) 
    elif config.model == "RecurrentGIN":
        model_config = config.RecurrentGIN_params
        model = RecurrentGIN(model_config.num_mlp_layers, input_size, model_config.hidden_dim, config.num_eigenvectors, model_config.global_dim, model_config.dropout, True, "Sum", device).to(device) # TODO: make these hyperparams configurable in command line
    elif config.model == "MLP":
        model_config = config.MLP_params
        model = MLP(model_config.num_layers, input_size, model_config.hidden_dim, config.num_eigenvectors).to(device)
    elif config.model == "MLP2":
        model_config = config.MLP2_params
        model = MLP2(model_config.num_layers, input_size, model_config.hidden_dim, config.num_eigenvectors, model_config.dropout).to(device)
    elif config.model == "GlobalGIN":
        model_config = config.GlobalGIN_params
        model = GlobalGIN(model_config.num_layers, model_config.num_mlp_layers, model_config.final_mlp_layers, input_size, model_config.hidden_dim, config.num_eigenvectors, config.evec_len, model_config.dropout, True, "Sum", model_config.use_attention,  device).to(device) 
    elif config.model == "harmonic":
        model_config = config.harmonic_params
        if model_config.subspace_size == None:
            model_config.subspace_size = 2 * config.num_eigenvectors + 1
        model = HarmonicAlgorithm(config.num_eigenvectors, model_config.subspace_size) 
    elif config.model == "lanczos":
        model_config = config.lanczos_params
        if model_config.subspace_size == None:
            model_config.subspace_size = 2 * config.num_eigenvectors + 1
        model = LanczosAlgorithm(config.num_eigenvectors, model_config.subspace_size)
    elif config.model == "ground_truth":
        model = GroundTruth(config.num_eigenvectors)
    elif config.model == "random_vectors":
        model = RandomVectors(config.num_eigenvectors)
    else:
        print("Invalid model type")
    

    
    if config.load_model != None: # TODO: make this flexible, maybe it checks for checkpoint_folder/load_model if just loading the path given by load_model doesn't work 
        print(f"Loading checkpoint: {config.load_model}")
        model.load_state_dict(torch.load(config.load_model, weights_only=True))

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(f"checkpoints/{config.checkpoint_folder}", exist_ok=True)

    if config.train: # only need if training. This throws error if trying to use an analytic method for training 
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)




    os.makedirs('plots', exist_ok=True)
    os.makedirs(f"plots/{config.checkpoint_folder}", exist_ok=True)


    if config.train:
        print("training...")
        if config.model == "GlobalMLP":
            global_training_loop(model, train_loader, val_loader, optimizer, device, config)
        else:
            training_loop(model, train_loader, val_loader, test_loader, optimizer, device, config)

    if config.test:
        print("testing...")
        evaluate(model, test_loader, device, config) # TODO: change to test_loader when ready for final analysis
        if not config.forced_ortho:
            evaluate(model, test_loader, device, config, extra_ortho_results=True)

    
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
