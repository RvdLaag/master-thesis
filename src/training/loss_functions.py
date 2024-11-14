import torch
import math

def _standard_normal_cdf(x, a=0, b=1):
    return (1+torch.special.erf((x-a)/(b*math.sqrt(2))))/2
def _trunc_cdf(x,mu,sig):
    xi = (x-mu)/sig
    alpha = -mu/sig
    cdf_alpha = _standard_normal_cdf(alpha)
    return (_standard_normal_cdf(xi)-cdf_alpha)/(1-cdf_alpha)

def _lognorm_cdf(x,mu,sig):
    #cdf = torch.zeros_like(x)
    cdf = _standard_normal_cdf((torch.log(x)-mu)/sig)
    return cdf


def CRPS_truncnorm_int(means: torch.Tensor, stds: torch.Tensor, target: torch.Tensor, crpsweight: bool|int = False, xdiff: int = 100, xn: int = 1000) -> torch.Tensor:
    
    
    xo_lower = torch.arange(-xdiff, 0, xdiff/xn, device=target.device).repeat(target.size(0),1).T
    xo_upper = torch.arange(0,xdiff, xdiff/xn, device=target.device).repeat(target.size(0),1).T
    x_lower = torch.add(xo_lower, target)
    x_upper = torch.add(xo_upper, target)

    y_lower = _trunc_cdf(x_lower[:,:,None], means, stds)**2
    y_upper = (_trunc_cdf(x_upper[:,:,None], means, stds)-1)**2
    if crpsweight:
        y_lower *= _standard_normal_cdf(xo_lower[:,:,None], b=crpsweight)
        y_upper *= _standard_normal_cdf(xo_upper[:,:,None], b=crpsweight)
    #                                                             v - was 1 before, should be -1 right?
    trap = torch.trapezoid(torch.cat((y_lower.T, y_upper.T), dim=-1), dx=xdiff/xn) # TODO: fix y_lower.T and y_upper.T | UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor.

    return trap

def CRPS_lognorm_int(means: torch.Tensor, stds: torch.Tensor, target: torch.Tensor, crpsweight: bool|int = False, xdiff: int = 100, xn: int = 1000) -> torch.Tensor:
    
    
    xo_lower = torch.arange(-xdiff, 0, xdiff/xn, device=target.device).repeat(target.size(0),1).T
    xo_upper = torch.arange(0,xdiff, xdiff/xn, device=target.device).repeat(target.size(0),1).T
    x_lower = torch.add(xo_lower, target)
    x_upper = torch.add(xo_upper, target)

    x_lower = torch.max(x_lower, torch.tensor([1e-10]).to(x_lower.device))
    x_upper = torch.max(x_upper, torch.tensor([1e-10]).to(x_upper.device))

    y_lower = _lognorm_cdf(x_lower[:,:,None], means, stds)**2
    y_upper = (_lognorm_cdf(x_upper[:,:,None], means, stds)-1)**2
    if crpsweight:
        y_lower *= _standard_normal_cdf(xo_lower[:,:,None], b=crpsweight)
        y_upper *= _standard_normal_cdf(xo_upper[:,:,None], b=crpsweight)
    #                                                             v - was 1 before, should be -1 right?
    trap = torch.trapezoid(torch.cat((y_lower.T, y_upper.T), dim=-1), dx=xdiff/xn) # TODO: fix y_lower.T and y_upper.T | UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor.

    return trap.float()

def CRPS_lognormsum_int(means: torch.Tensor, stds: torch.Tensor, target: torch.Tensor, crpsweight: bool|int = False, xdiff: int = 100, xn: int = 1000) -> torch.Tensor:
    

    xo_lower = torch.arange(-xdiff, 0, xdiff/xn, device=target.device).repeat(target.size(0),1).T
    xo_upper = torch.arange(0,xdiff, xdiff/xn, device=target.device).repeat(target.size(0),1).T
    x_lower = torch.add(xo_lower, target)
    x_upper = torch.add(xo_upper, target)

    y_lower = torch.zeros_like(x_lower[:,:,None])
    y_upper = torch.zeros_like(x_upper[:,:,None])
    for i in range(1,means.shape[1]+1):
        x_lower_max = torch.max(x_lower+i-1, torch.tensor([1e-10]).to(x_lower.device))
        x_upper_max = torch.max(x_upper+i-1, torch.tensor([1e-10]).to(x_upper.device))
        y_lower += (means.shape[1]-i+1)/(means.shape[1]*(means.shape[1]+1)/2)*_lognorm_cdf(x_lower_max[:,:,None], means[:,-i], stds[:,-i])**2
        y_upper += (means.shape[1]-i+1)/(means.shape[1]*(means.shape[1]+1)/2)*(_lognorm_cdf(x_upper_max[:,:,None], means[:,-i], stds[:,-i])-1)**2

    #y_lower = _lognorm_cdf(x_lower[:,:,None], means, stds)**2
    #y_upper = (_lognorm_cdf(x_upper[:,:,None], means, stds)-1)**2
    if crpsweight:
        y_lower *= _standard_normal_cdf(xo_lower[:,:,None], b=crpsweight)
        y_upper *= _standard_normal_cdf(xo_upper[:,:,None], b=crpsweight)
    #                                                             v - was 1 before, should be -1 right?
    trap = torch.trapezoid(torch.cat((y_lower.T, y_upper.T), dim=-1), dx=xdiff/xn) # TODO: fix y_lower.T and y_upper.T | UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor.

    return trap.float()


def CRPS_norm(y: torch.Tensor, var: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(var) * (y*torch.special.erf((y-mean)/torch.sqrt(2*var)) + 
                              math.sqrt(2/torch.pi)*torch.exp(-(y-1)**2 / (2*var)) - 1/math.sqrt(torch.pi))


def loss_std_bm(var1: torch.Tensor, var2: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(var2-var1)