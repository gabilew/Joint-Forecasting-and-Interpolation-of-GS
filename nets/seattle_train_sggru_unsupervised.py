import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import time

from data.Load_data import Seattle_data
from data.Dataloader import PrepareSampledDataset

from pytorch_gsp.train.train_rnn import  TestModel,  TrainModel
from pytorch_gsp.utils.gsp import ( greedy_e_opt, spectral_components)
from pytorch_gsp.models.sggru import *

def n_params(model):
    params=[]
    for param in model.parameters():
        params.append(param.numel())
    return np.sum(params)

print(torch.__version__)
device = 'cuda' if torch.cuda.is_available else 'cpu'
print("device is " + device)


def training_routine(args):
    lr = args.lr
    epochs = args.epochs
    pred_len = args.pred_len
    patience = args.patience
    name = args.save_name
    speed_matrix, A, FFR = Seattle_data('torch-gsp/data/Seattle_Loop_Dataset/')
    N = speed_matrix.shape[1]
    #speed_matrix  = speed_matrix.values
    S = int(args.sample_perc*N/100)
    if args.F_perc is None:
        F = int(S/3)
    else:
        F = int(args.F_perc*N/100)

    assert(S>F) # the sampling set must be larger than the spectral support

    #compute gft
    F_list, V = spectral_components(A,np.array(speed_matrix)[:100] )
    freqs = np.arange(0,F,1)

    if args.e_opt:
        print(args.sample_perc)
        if args.sample_perc == 25:
            sample = np.load( 'torch-gsp/data/Seattle_Loop_Dataset/sample_opt25.npy')[0]
        elif args.sample_perc == 50:
            sample = np.load( 'torch-gsp/data/Seattle_Loop_Dataset/sample_opt50.npy')[0]
        elif args.sample_perc == 75:
            sample = np.load( 'torch-gsp/data/Seattle_Loop_Dataset/sample_opt75.npy')[0]
        else:    
            sample = greedy_e_opt(V[:,Fs],S)
        
    else: sample = np.sort(np.random.choice(np.arange(N), S, replace = False))

    

    S = len(sample)        
    pre_time = time.time()
    train_dataloader, valid_dataloader, test_dataloader, max_speed = PrepareSampledDataset(speed_matrix, sample, 
                                                                                  V,freqs,T = True ,pred_len=pred_len, sampling='reduce')
                                                                  
    print("Preprocessing time:", time.time()-pre_time)
        


    layer = SpectralGraphForecast(V, sample,freqs, rnn = 'gru')
    sggru = model(V,sample,freqs, layer,l1=0,l2=0.0,supervised = False).to(device)
    
    pre_time = time.time()

    print("Total number of nodes: {}".format(N))
    print("Sample size: {}".format(S))
    print("Spectral sample size: {}".format(F))
    print("Initial learning rate: {}".format(lr))

    sggru,sggru_loss= TrainModel(sggru ,train_dataloader, valid_dataloader, num_epochs = epochs, learning_rate = lr,patience=patience,min_delta=1e-6, sample = sample)
    print("Training time:", time.time()-pre_time)
    pre_time = time.time()
    sggru_test = TestModel(sggru.to(device), test_dataloader, max_speed )
    print("Test time:", time.time()-pre_time)
    name = 'sggru'

    loss = (sggru_loss,sggru_test)
    os.makedirs("models_and_losses/", exsist_ok=True)
    #torch.save(sggru, "models_and_losses/{}.pt".format(name))
    #np.save("models_and_losses/{}.npy".format(name),loss)
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semi-Supervised Prediction\n SeattleLoop dataset ')
    parser.add_argument('--epochs', type=int, default = 100, help='maximum number of epochs before stopping training')
    parser.add_argument('--lr', type=float, default = 1e-4, help='starting learn rate' )
    parser.add_argument('--patience', type=int, default = 10, help='number of consecutive non-improving validation loss epochs before stop training')
    parser.add_argument('--sample-perc', type=int, default = 50, help='percentage of in-sample nodes')
    parser.add_argument('--F-perc', type=int, default = None, help='percentage of frequencies to keep in frequency set \mathcal{F}')
    parser.add_argument('--S-perc', type=int, default = 50, help='percentage of samples')
    parser.add_argument('--e-opt', action='store_true',help='if sampling is performed by E-optmal greedy algorithm')
    parser.add_argument('--sample-seed',type=int,default=1, help='number of run with uniformely random samples. Only used if --e-opt is False')
    parser.add_argument('--pred-len', type=int,default=1, help='prediction horizon')
    parser.add_argument('--save-name', type=str, default='sggru_S50_F17_opt_pred1', help='name of file')

    args = parser.parse_args()
    training_routine(args)


