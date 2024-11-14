import numpy as np

def _eps_lin(i, min_eps, max_eps, decay, bumprate, nbumps):
    if bumprate is not None:
        if nbumps is None or nbumps+1 > i/bumprate:
            i = i % bumprate
    return np.maximum(min_eps, max_eps - i*(max_eps-min_eps)*decay)

def _eps_exp(i, min_eps, max_eps, decay, bumprate, nbumps):
    if bumprate is not None:
        if nbumps is None or nbumps+1 > i/bumprate:
            i = i % bumprate
    return min_eps + (max_eps-min_eps)*np.exp(-i*decay)

def eps_decay(which='lin', decay=1/6000, bumprate=None, nbumps=None):
    if which == 'lin':
        return lambda i, min_eps, max_eps: _eps_lin(i, min_eps, max_eps, decay, bumprate, nbumps)
    elif which == 'exp':
        return lambda i, min_eps, max_eps: _eps_exp(i, min_eps, max_eps, decay, bumprate, nbumps)
    else:
        raise ValueError("which must be lin or exp")