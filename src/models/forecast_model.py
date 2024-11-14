from typing import Tuple

import torch
import torch.nn as nn

class BELU(nn.Module):
    def __init__(self, b: int = 10):
        super(BELU, self).__init__()
        self.elu = nn.ELU()
        self.b = b
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.min(self.elu(x)+1, torch.full_like(x, self.b+1)) # b+1 or not??
    
class TruncNormNetwork(nn.Module):
    def __init__(
        self, 
        datashape: Tuple[int, int], 
        hidden_dim: int = 32, 
        num_layers: int = 1, 
        dropout: int = 0, 
        belumax: int = 20
    ):
        super(TruncNormNetwork, self).__init__()

        self.windowsize = datashape[0]
        self.nsensors = datashape[1]

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.belumax = belumax

        #self.fc_sensors1 = nn.Linear(self.nsensors, self.nsensors-1) # apply to each timestep/column (or is it row?)
        #self.fc_settings1 = nn.Linear(3, 1) # allows for cluster of operational settings into 1 value per mode | need more?

        #self.lstm = nn.LSTM(self.nsensors, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm = nn.LSTM(self.nsensors, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, num_layers, batch_first=True, dropout=dropout)
        # | TEMP -> num_layers not implemented
        #self.fc_out1 = nn.Linear(hidden_dim//2, hidden_dim//2)

        self.fc_mean = nn.Linear(hidden_dim//2, 1)
        self.fc_std = nn.Linear(hidden_dim//2, 1)

        self.fc = nn.Linear(hidden_dim//2, 2)

        self.relu = nn.ReLU(inplace=False) # for truncnorm
        self.lrelu = nn.LeakyReLU(negative_slope=6, inplace=False) # for lognorm ?
        self.belu = BELU(b=belumax)
        self.tanh = nn.Tanh() # lognorm mu output

        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #o = self.relu(self.fc_settings1(x[:,:,:3])) # first 3 in final dim = operational settings
        #x = self.fc_sensors1(x) # last dim: nsensors -> nsensors-3

        #x = torch.cat([o,x], 2) # output N:windowsize:(nsensors-2) | add back clustered operational settings (mode)

        # h0 and c0 not needed unless want to provide non-zeros (LSTM defaults to h0,c0=zeros if not provided) | if providing h0 remember to .detach()
        #h0 = torch.randn(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to("cuda:0")
        #c0 = torch.randn(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to("cuda:0")
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(out)
        out, (hn, cn) = self.lstm2(out)

        #out = self.fc_out1(out[:,-1,:])
        # out = out[:,-1,:] # TRYING ALL HIDDEN OUTPUTS LSTM

        #out_means = self.relu(self.fc_mean(out))
        #out_stds = (self.belu(self.fc_std(out))+1)

        out = self.fc(out)
        #out_means = self.relu(out[:,0]).unsqueeze(1) # for truncnorm
        out_means = 6*self.tanh(out[...,0]).unsqueeze(-1) # for lognorm # TODO: DONT HARDCODE 6*
        #out_means = out[...,0].unsqueeze(-1)
        out_stds = self.belu(out[...,1]).unsqueeze(-1)
        #print(out_means.size())
        #print(out_stds.size())

        return out_means, out_stds