import sys
import os
# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.lines
import matplotlib.patheffects
import scipy.stats
import torch

from src.models.forecast_model import TruncNormNetwork
from src.data.processing import CMAPSS_dataset
from src.utils.test_funcs import *
from src.training.loss_functions import _standard_normal_cdf, _lognorm_cdf, CRPS_lognorm_int

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('dGreens', 
                                                               np.vstack([plt.cm.Greens(np.linspace(0,1,200)),
                                                                          plt.cm.Greens_r(np.linspace(0,1,200))]))

def _make_plot(ax, means, stds, ruls, rulspw, alpha):
    ax.invert_xaxis()
    ax.set_ylabel('RUL Prediction')
    ax.set_xlabel('True RUL')

    ax.plot(ruls, ruls, color='tab:red', linestyle='--', zorder=4)
    ax.plot(ruls, rulspw, color='tab:red', linestyle='-', zorder=4)
    
    # lows, ups = scipy.stats.truncnorm.interval(alpha, -means/stds, np.inf, loc=means, scale=stds)
    #alphas = np.arange(alpha, 0,-.1)[::-1]
    alphas = np.arange(alpha, 0.05-.01, -.01)[::-1]
    lows, ups = scipy.stats.lognorm.interval(alphas, s=stds[:,np.newaxis], scale=np.exp(means[:,np.newaxis]))
    means = np.exp(means + stds**2 /2)
    if len(means) == 1:
        ax.errorbar(ruls, means, yerr=np.array([np.abs(means-lows[:,0]),np.abs(ups[:,0]-means)]), capsize=3, color='k', marker='o')
        ylim = ax.get_ylim()
    else:
        ax.plot(ruls, means, color='k', lw=1)
        ax.fill_between(ruls, means, lows[:,0], color='tab:blue', alpha=1-(alphas[0]*(.95-.1)+.1), linewidth=0)
        ax.fill_between(ruls, means, ups[:,0], color='tab:blue', alpha=1-(alphas[0]*(.95-.1)+.1), linewidth=0)
        for i in range(alphas.shape[0]-1):
            ax.fill_between(ruls, lows[:,i], lows[:,i+1], color='tab:blue', alpha=1-(alphas[i+1]*(.95-.1)+.1), linewidth=0)
            ax.fill_between(ruls, ups[:,i], ups[:,i+1], color='tab:blue', alpha=1-(alphas[i+1]*(.95-.1)+.1), linewidth=0)
        ax.set_xlim(ruls[0], ruls[-1])
    
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid(zorder=-1)

    return ax

def _make_plot_2(ax, ax2, means, stds, ruls, rulspw, alpha):
    ax.invert_xaxis()
    ax.set_ylabel('RUL Prediction')
    ax.set_xlabel('True RUL')

    ax.plot(ruls, ruls, color='tab:orange', linestyle='--', zorder=4, label='True RUL')
    ax.plot(ruls, rulspw, color='tab:orange', linestyle='-', zorder=4, label='Piecewise Linear True RUL')
    
    # lows, ups = scipy.stats.truncnorm.interval(alpha, -means/stds, np.inf, loc=means, scale=stds)
    #alphas = np.arange(alpha, 0,-.1)[::-1]
    alphas = np.arange(alpha, 0.05-.01, -.01)[::-1]
    lows, ups = scipy.stats.lognorm.interval(alphas, s=stds[:,np.newaxis], scale=np.exp(means[:,np.newaxis]))

    plotruls = np.maximum(0,np.linspace(min(rulspw[-1]-5, lows[-1,-1]-np.abs(np.exp(means[-1] + stds[-1]**2 /2)-lows[-1,-1])*1.2), 
                                        max(rulspw[-1]+5, ups[-1,-1]+np.abs(ups[-1,-1]-np.exp(means[-1] + stds[-1]**2 /2))*1.2), 1000))
    plot_pdf = scipy.stats.lognorm.pdf(plotruls, s=stds[-1], scale=np.exp(means[-1]))
    means = np.exp(means + stds**2 /2)
    if len(means) == 1:
        ax.errorbar(ruls, means, yerr=np.array([np.abs(means-lows[:,0]),np.abs(ups[:,0]-means)]), capsize=3, color='k', marker='o')
        ylim = ax.get_ylim()
    else:
        ax.plot(ruls, means, color='k', lw=1)
        ax.fill_between(ruls, means, lows[:,0], color='tab:green', alpha=1-(alphas[0]*(.95-.1)+.1), linewidth=0)
        ax.fill_between(ruls, means, ups[:,0], color='tab:green', alpha=1-(alphas[0]*(.95-.1)+.1), linewidth=0)
        for i in range(alphas.shape[0]-1):
            ax.fill_between(ruls, lows[:,i], lows[:,i+1], color='tab:green', alpha=1-(alphas[i+1]*(.95-.1)+.1), linewidth=0)
            ax.fill_between(ruls, ups[:,i], ups[:,i+1], color='tab:green', alpha=1-(alphas[i+1]*(.95-.1)+.1), linewidth=0)
        ax.set_xlim(ruls[0], ruls[-1])
    
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid(zorder=-1)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([0],[0], color='k', label='Forecast 95% CI',path_effects=[matplotlib.patheffects.Stroke(linewidth=8, foreground=matplotlib.colors.to_rgba('tab:green',.5)),matplotlib.patheffects.Normal()]))
    ax.legend(handles=handles)

    ax2.set_xlabel('RUL')
    ax2.set_ylabel('Probability Density')
    
    ax2.plot(plotruls, plot_pdf, color='tab:green', zorder=4)
    ax2.axvline(means[-1], color='k', zorder=5, label='Mean Prediction', linewidth=2)
    ax2.axvline(rulspw[-1], color='tab:orange', linestyle='-', zorder=5, label='True RUL', linewidth=2)

    ax2.fill_between(plotruls, plot_pdf, where=np.logical_and(plotruls<=ups[-1,-1], plotruls>=lows[-1,-1]), color='tab:green', alpha=0.3, linewidth=0, zorder=2, label='95% CI')
    #ax2.fill_between(plotruls, plot_pdf, where=np.logical_and(plotruls>means[-1], plotruls<=ups[-1,0]), color='tab:blue', alpha=1-(alphas[0]*(.95-.1)+.1), linewidth=0)

    #for i in range(alphas.shape[0]-1):
        #ax2.fill_between(plotruls, plot_pdf, where=np.logical_and(plotruls<lows[-1,i], plotruls>=lows[-1,i+1]), color='tab:blue', alpha=1-(alphas[i+1]*(.95-.1)+.1), linewidth=0)
        #ax2.fill_between(plotruls, plot_pdf, where=np.logical_and(plotruls>ups[-1,i], plotruls<=ups[-1,i+1]), color='tab:blue', alpha=1-(alphas[i+1]*(.95-.1)+.1), linewidth=0)
    ax2.grid(zorder=-1)
    ax2.set_xlim(min(rulspw[-1]-5, lows[-1,-1]-np.abs(means[-1]-lows[-1,-1])*.2), 
                 max(rulspw[-1]+5, ups[-1,-1]+np.abs(ups[-1,-1]-means[-1])*.2))
    ax2.set_ylim(0,ax2.get_ylim()[1])
    ax2.legend(loc='upper right')
    return ax, ax2

def forecast_test_valengine(model: TruncNormNetwork, engineid: int, dataset: CMAPSS_dataset, alpha: float = 0.95, savepath: str|None = None, return_output: bool = False) -> None:
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, ruls, rulspw, _ = dataset.get_unit_by_id(engineid)

    data = data.float().to(device)
    #data = data[:,1:].float().to(device)
    (means, stds) = model(data)
    means = np.squeeze(means[:,-1].cpu().detach().numpy()) # only care about the last time step => [:,-1]
    stds = np.squeeze(stds[:,-1].cpu().detach().numpy())

    ruls = ruls.cpu().numpy()
    rulspw = rulspw.cpu().numpy()

    fig, (ax,ax2) = plt.subplots(1,2,layout='tight', figsize=(12,5), gridspec_kw={'width_ratios':[2,1]})
    ax.set_title(f'Val engine: {engineid} | {alpha*100:.0f}% CI')
    ax, ax2 = _make_plot_2(ax, ax2, means, stds, ruls, rulspw, alpha)
    plt.show()


def forecast_test_testset(model: TruncNormNetwork, testset: CMAPSS_dataset, alpha: float = 0.95, to_plot: int|bool = 5, savepath: str|None = None, return_metrics: bool = False) -> None:
    engineids = np.unique(testset.engineid.numpy())
    if isinstance(to_plot, bool):
        to_plot = len(engineids)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = {'RMSE': [],
               'MAE': [],
               'SF': [],
               'RMSE-median': [],
               'MAE-median': [],
               'SF-median': [],
               'PICP': [],
               'NMPIW': [],
               'CRPS': []}
    
    rulsmetrics = []
    mu_metrics = []
    sig_metrics = []

    for i, engineid in enumerate(engineids):
        data, ruls, rulspw, _ = testset.get_unit_by_id(engineid)
        data = data.float().to(device)
        
        #data = data[:,1:].float().to(device)
        (means, stds) = model(data)
        means = means[:,-1].cpu().detach().numpy()[:,0]#np.squeeze(means.cpu().detach().numpy())
        stds = stds[:,-1].cpu().detach().numpy()[:,0]#np.squeeze(stds.cpu().detach().numpy())
        
        ruls = ruls.cpu().numpy()
        rulspw = rulspw.cpu().numpy()

        mean_metric = np.array([np.exp(means[-1]+0.5*stds[-1]**2)]) # log norm
        median_metric = np.array([np.exp(means[-1])])
        std_metric = np.array([np.sqrt(np.exp(2*means[-1]+stds[-1]**2)*(np.exp(stds[-1]*2 -1)))]) # log norm
        
        mu_metrics.append(means[-1])
        sig_metrics.append(stds[-1])


        # TODO proper mean/std for truncnorm - see wiki or use scipy
        #mean_metric = np.array([means[-1] + stds[-1]*scipy.stats.norm.pdf(-means[-1]/stds[-1])/(1-scipy.stats.norm.cdf(-means[-1]/stds[-1]))])
        #std_metric = np.array([stds[-1]*np.sqrt(1-means[-1]/stds[-1]*scipy.stats.norm.pdf(-means[-1]/stds[-1])/(1-scipy.stats.norm.cdf(-means[-1]/stds[-1])) 
        #                                         - (scipy.stats.norm.pdf(-means[-1]/stds[-1])/(1-scipy.stats.norm.cdf(-means[-1]/stds[-1])))**2)])

        rul_metric = np.array([rulspw[-1]])
        rulsmetrics.append(rul_metric)
        # metrics only for the last window = what is done in literature
        # TODO median instead of mean? - sjoerd suggestion
        metrics['RMSE'].append(RMSE(mean_metric, rul_metric))
        metrics['MAE'].append(MAE(mean_metric, rul_metric))
        metrics['SF'].append(SF(mean_metric, rul_metric))
        metrics['RMSE-median'].append(RMSE(median_metric, rul_metric))
        metrics['MAE-median'].append(MAE(median_metric, rul_metric))
        metrics['SF-median'].append(SF(median_metric, rul_metric))
        metrics['PICP'].append(PICP((means[-1],stds[-1]), rul_metric))
        metrics['NMPIW'].append(NMPIW((means[-1],stds[-1]), rul_metric))
        metrics['CRPS'].append(CRPS_lognorm_int(torch.as_tensor(means[-1]).unsqueeze(0), torch.as_tensor(stds[-1]).unsqueeze(0), torch.as_tensor([rulspw[-1]])).item())

        if i < to_plot or savepath is not None:
            fig, (ax,ax2) = plt.subplots(1,2,layout='tight', figsize=(12,5), gridspec_kw={'width_ratios':[2,1]})
            ax.set_title(f'Test engine {engineid} | {alpha*100:.0f}% CI')
            ax, ax2 = _make_plot_2(ax, ax2, means, stds, ruls, rulspw, alpha)
            
            if savepath is not None:
                plt.savefig(f'{savepath}/testengine-{engineid}.png', facecolor='white', bbox_inches='tight')

            if i >= to_plot:
                plt.close()
            else:
                plt.show()
    metrics['NMPIW'] = [val/(np.max(rulsmetrics)-np.min(rulsmetrics)) for val in metrics['NMPIW']]

    
    
    
    return metrics, [rulsmetrics, mu_metrics, sig_metrics]


def calibration_tests(model, dataset, device):
    # Local helper functions
    def Ncdf(z, a=0, b=1):
        return 1/2 * (1+torch.erf((z-a)/(b*2**(-1/2))))
    def invNcdf(p, a=0, b=1):
        return a + b*torch.sqrt(torch.tensor(2))*torch.erfinv(2*p-1)
    def inverse_truncdf(x, mu, sig):
        return invNcdf(Ncdf(-mu/sig)+x*(1-Ncdf(-mu/sig)))*sig + mu
    def inverse_lognormcdf(x,mu,sig):
        #return torch.exp(mu + sig*invNcdf(x))
        return torch.exp(mu + torch.sqrt(torch.tensor(2))*sig*torch.erfinv(2*x-1))
    
    engineids = np.unique(dataset.engineid.numpy())
    
    quantiles = torch.linspace(0,1,11).unsqueeze(1)
    empcdf = torch.zeros(11)
    xs = torch.linspace(0,250,251).unsqueeze(1)
    Fbar = torch.zeros(251)
    Gbar = torch.zeros(251)
    PIT_vals = []
    PIT_ruls = []

    n = 0
    for i, engineid in enumerate(engineids):
        data, ruls, rulspw, _ = dataset.get_unit_by_id(engineid)
        data = data.float().to(device)
        (mu, sig) = model(data)
        mu = np.squeeze(mu[:,-1].cpu().detach())
        sig = np.squeeze(sig[:,-1].cpu().detach())

        ruls = ruls.cpu()
        rulspw = rulspw.cpu()

        PIT_vals.append(_lognorm_cdf(rulspw, mu, sig)[-1])
        PIT_ruls.append(rulspw[-1])

        icdfvals = inverse_lognormcdf(quantiles, mu, sig)
        empcdf += torch.sum(rulspw <= icdfvals, dim=1)

        Fbar += torch.sum(_lognorm_cdf(xs, mu, sig), dim=1)
        Gbar += torch.sum(xs >= rulspw, dim=1)

        try:
            n += len(mu)
        except TypeError:
            if mu.dim() == 0:
                n += 1

    empcdf /= n
    Fbar /= n
    Gbar /= n 
    PIT_vals = np.asarray(PIT_vals)
    PIT_ruls = np.asarray(PIT_ruls)

    return quantiles, empcdf, xs, Fbar, Gbar, PIT_vals, PIT_ruls

