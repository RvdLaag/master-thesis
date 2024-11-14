import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor,
        cvar_alpha: float = 1.0 # in [0,1] | 1 => mean
    ):
        super(Network, self).__init__()

        if isinstance(in_dim, (tuple, list, np.ndarray, torch.Tensor)):
            in_dim = int(np.prod(in_dim))
        assert isinstance(in_dim, int)
            

        self.alpha = cvar_alpha

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.layers = nn.Sequential(
            #nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, out_dim * atom_size) # TODO try module list to seperate distributions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        if self.alpha == 1:
            q = torch.sum(dist * self.support, dim=2)
        else:
            q = self._calc_CVaR(dist)
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        #x = torch.flatten(x, start_dim=-2, end_dim=-1)
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist
    
    def _calc_CVaR(self, dist):
        """Compute CVaR for discrete distributions"""
        C = torch.cumsum(dist, dim=-1)
        C[...,-1] = 1 # rounding errors fix/prevention
        k = torch.argmax((C>=self.alpha).to(int), dim=-1)
        rangetensor = torch.arange(dist.size(-1), device=dist.device).expand(dist.size())
        indexmask = rangetensor <= k.unsqueeze(-1)
        CVaR = torch.sum(dist*self.support*indexmask, dim=-1)/torch.sum(dist*indexmask, dim=-1)
        return CVaR