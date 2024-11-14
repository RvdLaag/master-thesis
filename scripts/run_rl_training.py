import sys
import os
# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parent_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))


import argparse

import numpy as np
import torch

from src.models.forecast_model import TruncNormNetwork
from src.utils.environment import ForecastEnv
from src.models.rl_agent import DQNAgent

def seed_torch(seed): # TODO: place elsewhere ?
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_rl(forecastmodel, dataset, engines, test_engines, seed, n_actions, planning_window, n_obs, num_frames, memory_size, batch_size, target_update, epsilon_func, vmin, vmax, atom_size, window_size, cvar_alpha=1, plot_interval=200, cf=4, c1=0.1, c0=2, cn=0.1):
    np.random.seed(seed)
    seed_torch(seed)
    env = ForecastEnv(n_actions, planning_window, n_obs, forecastmodel, dataset, engines, test_engines, window_size, seed, cf, c1, c0, cn)

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_func, seed, v_min=vmin, v_max=vmax, atom_size=atom_size, cvar_alpha=cvar_alpha)

    agent.train(num_frames, to_plot=bool(plot_interval), plotting_interval=plot_interval)


    return agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=[1,2,3,4])
    parser.add_argument('windowsize', type=int, default=19)
    parser.add_argument('upperRUL', type=int, default=128)
    parser.add_argument('-n', '--numframes', type=int, default=10000)
    parser.add_argument('-m', '--memsize', type=int, default=2000)
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    parser.add_argument('-tu', '--targetupdate', type=int, default=150)
    parser.add_argument('-ed', '--epsdecay', type=float, default=1/2000)
    parser.add_argument('-s', '--seed', type=int, default=19960809)
    parser.add_argument('-id', '--identifier', type=str, default='')

    # Cat. DQN arguments
    parser.add_argument('-vmin', type=float, default=-10)
    parser.add_argument('-vmax', type=float, defaut=5)
    parser.add_argument('-atomsize', type=int, default=30)
    parser.add_argument('-nactions', type=int, default=11)
    parser.add_argument('-nobs', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=1) # 1 == use mean not cvar

    args = parser.parse_args()

    
    dataset = torch.load(parent_dir+f'/data/processed/trainsetFD00{args.dataset}_w{args.windowsize}_M{args.upperRUL}.pt')
    engines = torch.load(parent_dir+f'/data/used/rl_engines_{dataset}{args.identifier}.pt').numpy()

    forecastmodel = torch.load(parent_dir+f'/models/forecaster_{dataset}{args.identifier}.pt')

    agent = train_rl(forecastmodel, dataset, engines, 
                     seed=args.seed, n_actions=args.nactions, n_obs=args.nobs, num_frames=args.numframes, 
                     memory_size=args.memsize, batch_size=args.batchsize, target_update=args.targetupdate, 
                     epsilon_decay=args.epsdecay, vmin=args.vmin, vmax=args.vmax, atom_size=args.atomsize, 
                     cvar_alpha=args.alpha, plot_interval=0) # no update plotting if ran from command line
    
    torch.save(agent, parent_dir+f'/models/rlagent_{dataset}{args.identifier}.pt')

