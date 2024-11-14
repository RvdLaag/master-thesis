from typing import List, Dict

import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm

from src.training.loss_functions import CRPS_lognormsum_int # TEMP

def train(model, optim, loss_func, device, train_loader, val_loader, RULtype: str, num_epochs: int) -> Dict[str,List]:
    
    loaders = {'train': train_loader, 
               'val': val_loader}
    losses = {'train': [], 
              'val': []}
    
    for epoch in (pbar := tqdm(range(1, num_epochs+1), desc=f"Epoch: {0}/{num_epochs} | Train loss: {0} | Val loss {0}")):
        for phase in ['train', 'val']:
            phase_losses = []
            for batch_data, batch_target, batch_targetpw, _ in loaders[phase]:
                batch_data = batch_data.float().to(device)

                if RULtype.lower() == 'piecewise':
                    batch_target = batch_targetpw.to(device)
                else:
                    batch_target = batch_target.to(device)
                
                optim.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(batch_data)
                    loss = loss_func(*preds, batch_target)
                    loss = loss.mean()
                    if phase == 'train':
                        loss.backward()
                        optim.step()
                
                phase_losses.append(loss.cpu().detach().numpy())
            
            losses[phase].append(np.mean(phase_losses))
        
        pbar.set_description(f'Epoch: {epoch}/{num_epochs} | Train loss: {losses["train"][-1]:.2f} | Val loss: {losses["val"][-1]:.2f}')

    return losses

def train_bm(model, optim, loss_funcs, lambdas, device, train_loader, val_loader, upper_RUL: int, num_epochs: int) -> Dict[str,List]:
    RULtype = 'piecewise' if upper_RUL>0 else 'linear'
    loaders = {'train': train_loader, 
               'val': val_loader}
    losses = {'train': [], 
              'val': []}
    
    for epoch in (pbar := tqdm(range(1, num_epochs+1), desc=f"Epoch: {0}/{num_epochs} | Train loss: {0} | Val loss {0}")):
        for phase in ['train', 'val']:
            phase_losses = []

            for batch_data, batch_target, batch_targetpw, _ in loaders[phase]:
                batch_data = batch_data.float().to(device)

                if RULtype.lower() == 'piecewise':
                    batch_target = batch_targetpw.to(device)
                    batch_bm = torch.ones_like(batch_target, dtype=torch.float32).to(device)
                    batch_bm[batch_target>=upper_RUL] = 0

                    batch_target = torch.stack([torch.arange(tar+model.windowsize-1, tar-1, -1, device=batch_target.device) for tar in batch_target])
                    batch_target = torch.minimum(batch_target, torch.tensor(upper_RUL))
                    
                else:
                    batch_target = batch_target.to(device)
                    batch_bm = torch.ones_like(batch_target, dtype=torch.float32).to(device)
                    batch_target = torch.stack([torch.arange(tar+model.windowsize-1, tar-1, -1, device=batch_target.device) for tar in batch_target])
                    

                
                optim.zero_grad()
                #with torch.no_grad():
                    
                with torch.set_grad_enabled(phase == 'train'):
                    #prev_preds = model(batch_data[:,:-1])
                    #preds = model(batch_data[:,1:])
                    preds = model(batch_data)
                    
                    # TODO also loss_main for prev_preds ????
                    loss_main = loss_funcs[0](*(torch.flatten(preds[0],0,1), torch.flatten(preds[1],0,1)), torch.flatten(batch_target,0,1))
                    loss_main = torch.unflatten(loss_main, 1, (preds[0].shape[:-1])) * torch.arange(1,preds[0].shape[1]+1, device=loss_main.device)/(preds[0].shape[1]*(preds[0].shape[1]+1)/2)
                    loss_main = loss_main.mean()
                    
                    #loss_means = loss_funcs[1](prev_preds[0]-preds[0] + prev_preds[0]*torch.exp(torch.distributions.normal.Normal(0,1).log_prob(-prev_preds[0]/prev_preds[1]))/(1-torch.distributions.normal.Normal(0,1).cdf(-prev_preds[0]/prev_preds[1]))
                    #                           - preds[0]*torch.exp(torch.distributions.normal.Normal(0,1).log_prob(-preds[0]/preds[1]))/(1-torch.distributions.normal.Normal(0,1).cdf(-preds[0]/preds[1])), 
                    #                           batch_bm.unsqueeze(1))# FOR TRUNCNORM
                    #loss_means = loss_funcs[1](torch.exp(prev_preds[0] + 0.5*prev_preds[1]**2) - torch.exp(preds[0] + 0.5*preds[1]**2 ), batch_bm.unsqueeze(1)) # FOR LOGNORM | mean
                    #loss_means = loss_funcs[1](torch.exp(prev_preds[0]) - torch.exp(preds[0]), batch_bm.unsqueeze(1)) # FOR LOGNORM | median
                    #loss_means = loss_funcs[1](torch.diff(torch.exp(preds[0] + 0.5*preds[1]**2),dim=-2).squeeze(), torch.diff(batch_target))
                    #loss_means = loss_funcs[1](torch.exp(prev_preds[0][:,-1] + 0.5*prev_preds[1][:,-1]**2) - torch.exp(preds[0][:,-1] + 0.5*preds[1][:,-1]**2 ), batch_bm.unsqueeze(1)) # FOR LOGNORM | mean
                    #loss_means = loss_funcs[1](torch.exp(prev_preds[0][:,-1]) - torch.exp(preds[0][:,-1]), batch_bm.unsqueeze(1)) # FOR LOGNORM | median
                    
                    #loss_means = CRPS_lognormsum_int(preds[0], preds[1], batch_target[:,1])
                    loss_means = torch.tensor([0.], device=loss_main.device)
                    
                    loss_means = loss_means.mean().float()
                    
                    # loss_var = loss_funcs[2](prev_preds[1]**2, preds[1]**2)*batch_bm
                    # loss_var = loss_var.mean()

                    loss = loss_main + lambdas*loss_means
                    
                    if phase == 'train':
                        loss.backward()
                        optim.step()

                phase_losses.append([loss_main.cpu().detach().numpy(), loss_means.cpu().detach().numpy()])
            phase_losses = np.asarray(phase_losses)
            losses[phase].append([np.mean(phase_losses[:, 0]), np.mean(phase_losses[:, 1])])
        
        pbar.set_description(f'Epoch: {epoch}/{num_epochs} | Train loss: {losses["train"][-1][0]:.2f}, {losses["train"][-1][1]:.2f} | Val loss: {losses["val"][-1][0]:.2f}, {losses["val"][-1][1]:.2f}')

    return losses