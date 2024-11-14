import sys
import os
# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parent_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functools import partial

from src.models.forecast_model import TruncNormNetwork
from src.data.processing import CMAPSS_dataset
from src.training.loss_functions import CRPS_truncnorm_int, CRPS_norm, loss_std_bm, CRPS_lognorm_int
from src.training.train_forecast_model import train, train_bm

def train_forecast(dataset, identifier, num_epochs, batch_size, learning_rate, hidden_dim, dropout, num_lstms, std_max, train_fraction, val_fraction, upper_RUL, window_size):
    
    traindata = torch.load(parent_dir+f'/data/processed/trainset{dataset}_w{window_size}_M{upper_RUL}.pt')
    use_engines, rl_engines = torch.utils.data.random_split(traindata.engineid.unique(), [train_fraction, 1-train_fraction])

    torch.save(rl_engines.dataset[rl_engines.indices], parent_dir+f'/data/used/rl_engines_{dataset}{identifier}.pt')

    train_engines, val_engines = torch.utils.data.random_split(use_engines, [1-val_fraction, val_fraction])

    train_set = torch.utils.data.Subset(traindata, np.where(np.in1d(traindata.engineid, train_engines.dataset[train_engines.indices]))[0])
    val_set = torch.utils.data.Subset(traindata, np.where(np.in1d(traindata.engineid, val_engines.dataset[val_engines.indices]))[0])

    dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TruncNormNetwork((traindata.data.shape[1],traindata.data.shape[2]), 
                             hidden_dim, 
                             num_lstms, 
                             dropout, 
                             std_max).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = train(model=model, 
                   optim=optim,
                   loss_func=CRPS_truncnorm_int, 
                   device=device, 
                   train_loader=dataloaders['train'], 
                   val_loader=dataloaders['val'], 
                   RULtype='piecewise' if upper_RUL>0 else 'linear',
                   num_epochs=num_epochs)


    return model, losses, optim, dataloaders

def train_forecast_bm(dataset, identifier, num_epochs, batch_size, learning_rate, hidden_dim, dropout, num_lstms, std_max, train_fraction, val_fraction, upper_RUL, window_size, crpsweight=None, lambd=.1, dataloaders=None):
    traindata = torch.load(parent_dir+f'/data/processed/trainset{dataset}_w{window_size}_M{upper_RUL}.pt')
    if dataloaders is None:
        use_engines, rl_engines = torch.utils.data.random_split(traindata.engineid.unique(), [train_fraction, 1-train_fraction])

        torch.save(rl_engines.dataset[rl_engines.indices], parent_dir+f'/data/used/rl_engines_{dataset}{identifier}.pt')

        train_engines, val_engines = torch.utils.data.random_split(use_engines, [1-val_fraction, val_fraction])

        train_set = torch.utils.data.Subset(traindata, np.where(np.in1d(traindata.engineid, train_engines.dataset[train_engines.indices]))[0])
        val_set = torch.utils.data.Subset(traindata, np.where(np.in1d(traindata.engineid, val_engines.dataset[val_engines.indices]))[0])

        dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
        }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TruncNormNetwork((traindata.data.shape[1], traindata.data.shape[2]), # width - 1 <- for BM testing
                             hidden_dim, 
                             num_lstms, 
                             dropout, 
                             std_max).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #lf = CRPS_truncnorm_int if crpsweight is None else partial(CRPS_truncnorm_int, crpsweight=crpsweight)
    lf = CRPS_lognorm_int if crpsweight is None else partial(CRPS_lognorm_int, crpsweight=crpsweight)
    losses = train_bm(model=model, 
                      optim=optim,
                      loss_funcs=[lf, torch.nn.MSELoss(reduction='sum')], #[CRPS_truncnorm_int, CRPS_norm, loss_std_bm], 
                      lambdas=lambd,
                      device=device,
                      train_loader=dataloaders['train'], 
                      val_loader=dataloaders['val'], 
                      upper_RUL=upper_RUL, # 'piecewise' if upper_RUL>0 else 'linear',
                      num_epochs=num_epochs)


    return model, losses, optim, dataloaders



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=[1,2,3,4])
    parser.add_argument('-n', '--numepochs', type=int, default=100)
    parser.add_argument('-b', '--batchsize', type=int, default=64)
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-3)
    parser.add_argument('-h', '--hiddendim', type=int, default=32)
    parser.add_argument('-d', '--dropout', type=float, default=0.0)
    parser.add_argument('-lstms', '--numlstms', type=int, default=2)
    parser.add_argument('-sm', '--stdmax', type=float, default=20)
    parser.add_argument('-tf', '--trainsetfraction', type=float, default=0.5)
    parser.add_argument('-vf', '--valsetfraction', type=float, default=0.1)
    parser.add_argument('-ur', '--upperRUL', type=int, default=128, help="0 if linear RUL target")
    parser.add_argument('-w', '--windowsize', type=int, default=19)
    parser.add_argument('-id', '--identifier', type=str, default='')


    args = parser.parse_args()

    dataset = f'FD00{args.datasets}'
    num_epochs = args.numepochs
    batch_size = args.batchsize
    learning_rate = args.learningrate
    hidden_dim = args.hiddendim
    dropout = args.dropout
    num_lstms = args.numlstms
    std_max = args.stdmax
    train_fraction = args.trainsetfraction
    val_fraction = args.valsetfraction # of train fraction
    upper_RUL = args.upperRUL
    window_size = args.windowsize
    identifier = args.identifier

    model, losses = train_forecast(dataset, identifier, num_epochs, batch_size,learning_rate, hidden_dim, dropout, num_lstms, std_max, train_fraction, val_fraction, upper_RUL, window_size)

    print(f"Saving loss figure to {f'../results/figures/losses_{dataset}{identifier}.png'}")
    f, ax = plt.subplots(layout='tight')
    ax.plot(range(1,num_epochs+1), losses['train'], label='Train loss')
    ax.plot(range(1,num_epochs+1), losses['val'], label='Val loss')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.savefig(parent_dir+f'/results/figures/losses_{dataset}{identifier}.png', bbox_inchex='tight', facecolor='white', dpi=100)
    
    print(f"Saving model to {f'../models/forecaster_{dataset}{identifier}.pt'}")
    torch.save(model, parent_dir+f'/models/forecaster_{dataset}{identifier}.pt')

    print('Finished')
