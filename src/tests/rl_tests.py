import sys
import os
# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Dict, List

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import scipy.stats
from IPython import display

from src.models.rl_agent import DQNAgent

def rl_test_engine_trunc(agent: DQNAgent, engineid: int, savepath: str|None = None, return_output: bool = False) -> None:
    assert engineid >= 0 and engineid < len(agent.env.test_engines), f"Engine id {engineid} out of range. {len(agent.env.test_engines)} engines available for testing."
    
    # Predict all data of selected engine
    agent.is_test = True
    engine = agent.env.test_engines[engineid]
    #states, ruls = agent.env.test_states(engineid)
    #with torch.no_grad():
        #actions = agent.dqn(states).argmax(1).cpu().numpy()
        #dists = agent.dqn.dist(states).cpu().numpy()
    #rewards, terminated = agent.env.test_rewards(actions, ruls)

    #states = states.cpu().numpy()

    states, ruls, rewards, actions, dists, cvars = agent.test(engineid)



    xs = np.linspace(0, ruls[0]+20, 300)
    cmap = plt.get_cmap('tab10')
    zj = agent.support.cpu().numpy()
    delta_z = float(agent.v_max - agent.v_min) / (agent.atom_size - 1)

    fig = plt.figure(layout='constrained', figsize=(8,5))

    subfigs = fig.subfigures(1,2, wspace=0.02, width_ratios=[2,1])
    ax = subfigs[0].subplots()
    
    ax.set_title(f'Engine: {engine} | Reward: {rewards[0]:.2f} | score: {np.sum(rewards[:0+1]):.2f}')
    mean, std = states[0,0], states[0,1]
    ydata = scipy.stats.truncnorm.pdf(xs, -mean/std, np.inf, loc=mean, scale=std)
    pdf_plot, = ax.plot(xs, ydata)
    ax.set_ylim(0, ydata.max()*1.1)
    rul_line = ax.axvline(ruls[0], color='k', linestyle='--')
    rul_text = ax.text(ruls[0], 0.99, f' RUL={ruls[0]:.0f}', color='k', ha='left', va='top', transform=ax.get_xaxis_transform())
    ax.set_xlabel('RUL')

    axd = subfigs[1].subplots(dists.shape[1], sharex=True)
    axd[0].set_title(f'replace at k={actions[0]}' if actions[0] != 0 else f'do nothing')
    bars = []
    meanlines = []
    ylabels = []
    for i in range(dists.shape[1]):
        axd[i].set_yticks([])
        ylabels.append(axd[i].text(-0.02, 0.5, f'replace {i:02d}' if i!=0 else f'nothing', transform=axd[i].transAxes, ha='right', va='center', color='g' if actions[0]==i else 'k', weight='bold' if actions[0]==i else None))
        c = 'k' if i == 0 else cmap((i-1)/10)
        bars.append(axd[i].bar(zj, dists[0,i], width=delta_z, align='center', color=c, edgecolor='w'))
        meanlines.append(axd[i].axvline(sum(zj*dists[0,i]), color='k' if i!=0 else 'r', linestyle='--'))
    
    def animate(i):
        mean, std = states[i,0], states[i,1]
        ydata = scipy.stats.truncnorm.pdf(xs, -mean/std, np.inf, loc=mean, scale=std)
        ax.set_ylim(0, ydata.max()*1.1)
        pdf_plot.set_ydata(ydata)
        rul_line.set_xdata(ruls[i])
        rul_text.set(position=(ruls[i],0.99), transform=ax.get_xaxis_transform(), text=f' RUL={ruls[i]:.0f}')
        if i < len(actions):
            ax.set_title(f'Engine: {engine} | Reward: {rewards[i]:.2f} | score: {np.sum(rewards[:i+1]):.2f}')
            axd[0].set_title(f'replace at k={actions[i]}' if actions[i] != 0 else f'do nothing')
            for j, (barplot, meanline) in enumerate(zip(bars, meanlines)):
                ylabels[j].set_color('g' if actions[i]==j else 'k')
                ylabels[j].set_fontweight('bold' if actions[i]==j else None)
                for k, bar in enumerate(barplot):
                    bar.set_height(dists[i,j,k])
                meanline.set_xdata(sum(zj*dists[i,j]))
        else:
            ax.set_title(f'Engine: {engine} | Final score: {np.sum(rewards):.2f}')
            axd[0].set_title(f'Terminated')
        return [pdf_plot, rul_line, rul_text] + bars + meanlines + ylabels
    
    ani = animation.FuncAnimation(fig, animate, frames=ruls.shape[0]-1, interval=100, blit=False, repeat=False)
    if savepath is None:
        vid = ani.to_html5_video()
        html = display.HTML(vid)
        display.display(html)
    else:
        writer = animation.FFMpegWriter(fps=2) 
        ani.save(f'{savepath}/{engine}-testani.mp4', writer=writer)
    plt.close()

    if return_output:
        return states, rewards
    
def rl_test_grid(agent: DQNAgent, minMu: float, maxMu: float, nMu: int, minSig: float, maxSig: float, nSig: int, savepath: str|None = None):
    assert minSig > 0
    Mus = torch.linspace(minMu, maxMu, nMu, dtype=torch.float32)
    Sigs = torch.linspace(minSig, maxSig, nSig, dtype=torch.float32)

    grid_mu, grid_sig = torch.meshgrid(Mus, Sigs, indexing='ij')
    pre_states = torch.vstack([grid_mu.flatten(), grid_sig.flatten()]).T.to(agent.device).unsqueeze(1)
    states = agent.env._transform_states(pre_states)

    with torch.no_grad():
        eval = agent.dqn(states)
        action = eval.argmax(1).cpu().numpy()
        cvar = eval.detach().cpu().numpy()
        dist = agent.dqn.dist(states).cpu().numpy()
    
    cmap = matplotlib.colors.ListedColormap(['k']+list(plt.get_cmap('tab10').colors))
    fig, ax = plt.subplots()
    im = ax.pcolormesh(np.exp(grid_mu), grid_sig, action.reshape((Mus.shape[0],Sigs.shape[0])), cmap=cmap, vmin=0, vmax=10)
    ax.set_xlim(0,128)
    ax.set_xlabel(r'$\mathrm{exp}(\mu)$')
    ax.set_ylabel(r'$\sigma$')
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0,11,1)+.5)
    cbar.set_ticklabels(['do nothing']+[f'replace {i:02d}' for i in range(1,11)])



def rl_test_grid(agent: DQNAgent, minRUL: int, maxRUL: int, minSTD: float, maxSTD: float, nSTD: int = 20, savepath: str|None = None):
    RULs = torch.arange(minRUL, maxRUL+1, 1, dtype=torch.float32)
    STDs = torch.linspace(minSTD, maxSTD, nSTD, dtype=torch.float32)

    grid_rul, grid_std = torch.meshgrid(RULs, STDs, indexing='ij')
    states = torch.vstack([grid_rul.flatten(), grid_std.flatten()]).T
    states = states.to(agent.device)

    with torch.no_grad(): # computes dists twice TODO fix ?
        CVaRs = agent.dqn(states)
        actions = CVaRs.argmax(1).cpu().numpy()
        CVaRs = CVaRs.cpu().numpy()
        dists = agent.dqn.dist(states).cpu().numpy()
    states = states.cpu().numpy()

    RULs = RULs.numpy()
    STDs = STDs.numpy()
    actions = actions.reshape((RULs.shape[0], STDs.shape[0]))
    CVaRs = CVaRs.reshape((RULs.shape[0], STDs.shape[0], CVaRs.shape[1]))
    dists = dists.reshape((RULs.shape[0], STDs.shape[0], dists.shape[1], dists.shape[2]))

    if dists.shape[2] == 11:
        cmap = matplotlib.colors.ListedColormap(['k']+list(plt.get_cmap('tab10').colors))
        norm = None
    else:
        cmap = plt.get_cmap('viridis')
        norm = matplotlib.colors.BoundaryNorm(np.arange(dists.shape[2]), cmap.N)
    
    fig, ax = plt.subplots()
    grid_ruls, grid_stds = np.meshgrid(RULs, STDs, indexing='ij')
    im = ax.pcolormesh(grid_ruls, grid_stds, actions, cmap=cmap, vmin=0, vmax=dists.shape[2], norm=norm)
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\sigma$')
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0,dists.shape[2],1)+.5)
    cbar.set_ticklabels(['do nothing']+[f'replace {i:02d}' for i in range(1,dists.shape[2])])
    if savepath is not None:
        plt.savefig(f'{savepath}/grid-action.png', bbox_inches='tight', facecolor='white')
    plt.show()
    
    fig, ax = plt.subplots(int(np.ceil(dists.shape[2]/4)), 4, layout='constrained', figsize=(12,8))
    ax = ax.flatten()
    vmin = agent.v_min
    vmax = agent.v_max
    for i in range(ax.shape[0]):
        if i < dists.shape[2]:
            ax[i].set_title('do nothing' if i == 0 else f'k={i:02d}')
            ax[i].set_xlabel('$\mu$')
            ax[i].set_ylabel('$\sigma$')
            im = ax[i].pcolormesh(grid_ruls, grid_stds, CVaRs[:,:,i], vmin=vmin, vmax=vmax, cmap='turbo')
        else: 
            ax[i].axis('off')
    fig.colorbar(im, ax=ax.ravel().tolist(), label='CVaR')
    if savepath is not None:
        plt.savefig(f'{savepath}/cvar-action.png', bbox_inches='tight', facecolor='white')
    plt.show()




    if False:
        return RULs, STDs, states, actions, dists









def rl_test_engine(agent: DQNAgent, engineid: int, savepath: str|None = None, return_output: bool = False) -> None:
    assert engineid >= 0 and engineid < len(agent.env.test_engines), f"Engine id {engineid} out of range. {len(agent.env.test_engines)} engines available for testing."
    
    # Predict all data of selected engine
    agent.is_test = True
    engine = agent.env.test_engines[engineid]
    #states, ruls = agent.env.test_states(engineid)
    #with torch.no_grad():
        #actions = agent.dqn(states).argmax(1).cpu().numpy()
        #dists = agent.dqn.dist(states).cpu().numpy()
    #rewards, terminated = agent.env.test_rewards(actions, ruls)

    #states = states.cpu().numpy()

    states, statesCDF, ruls, rewards, actions, dists, cvars = agent.test(engineid)
    


    xs = np.linspace(0, ruls[0]+20, 300)
    cmap = plt.get_cmap('tab10')
    zj = agent.support.cpu().numpy()
    delta_z = float(agent.v_max - agent.v_min) / (agent.atom_size - 1)

    fig = plt.figure(layout='constrained', figsize=(8,5))

    subfigs = fig.subfigures(1,2, wspace=0.02, width_ratios=[2,1])
    ax = subfigs[0].subplots()
    axh = ax.twinx().twiny()
    #axh.invert_yaxis()
    patches = axh.bar(np.arange(1,statesCDF.shape[1]+1), statesCDF[0,:], edgecolor='k', alpha=0.5)
    for i in range(statesCDF.shape[1]):
        patches[i].set_facecolor(cmap((i)/10))
    axh.set_ylim(1,0)
    ax.set_title(f'Engine: {engine} | Reward: {rewards[0]:.2f} | score: {np.sum(rewards[:0+1]):.2f}')
    mean, std = states[0,-1,0], states[0,-1,1]
    ydata = scipy.stats.lognorm.pdf(xs, s=std, scale=np.exp(mean))
    #ydata = np.zeros_like(xs)
    #for j, (mu, sig) in enumerate(zip(states[0,:,0][::-1], states[0,:,1][::-1])):
    #    ydata += (states.shape[1]-j)**2 /(states.shape[1]*(states.shape[1]+1)*(2*states.shape[1]+1)/6)*scipy.stats.lognorm.pdf(xs+j, s=sig, scale=np.exp(mu))
    pdf_plot, = ax.plot(xs, ydata)
    ax.set_ylim(0, ydata.max()*1.1)
    rul_line = ax.axvline(ruls[0], color='k', linestyle='--')
    rul_text = ax.text(ruls[0], 0.99, f' RUL={ruls[0]:.0f}', color='k', ha='left', va='top', transform=ax.get_xaxis_transform())
    ax.set_xlabel('RUL')

    axd = subfigs[1].subplots(dists.shape[1], sharex=True)
    axd[0].set_title(f'replace at k={actions[0]}' if actions[0] != 0 else f'do nothing')
    bars = []
    meanlines = []
    ylabels = []
    for i in range(dists.shape[1]):
        axd[i].set_yticks([])
        ylabels.append(axd[i].text(-0.02, 0.5, f'replace {i:02d}' if i!=0 else f'nothing', transform=axd[i].transAxes, ha='right', va='center', color='g' if actions[0]==i else 'k', weight='bold' if actions[0]==i else None))
        c = 'k' if i == 0 else cmap((i-1)/10)
        bars.append(axd[i].bar(zj, dists[0,i], width=delta_z, align='center', color=c, edgecolor='w'))
        meanlines.append(axd[i].axvline(sum(zj*dists[0,i]), color='k' if i!=0 else 'r', linestyle='--'))
    
    def animate(i):
        #i+=100 # REMOVE
        for j in range(statesCDF.shape[1]):
            patches[j].set_height(statesCDF[i,j])
        mean, std = states[i,-1,0], states[i,-1,1]
        ydata = scipy.stats.lognorm.pdf(xs, s=std, scale=np.exp(mean))
        #ydata = np.zeros_like(xs)
        #for j, (mu, sig) in enumerate(zip(states[i,:,0][::-1], states[i,:,1][::-1])):
        #    ydata += (states.shape[1]-j)**2 /(states.shape[1]*(states.shape[1]+1)*(2*states.shape[1]+1)/6)*scipy.stats.lognorm.pdf(xs+j, s=sig, scale=np.exp(mu))
        ax.set_ylim(0, ydata.max()*1.1)
        pdf_plot.set_ydata(ydata)
        rul_line.set_xdata(ruls[i])
        rul_text.set(position=(ruls[i],0.99), transform=ax.get_xaxis_transform(), text=f' RUL={ruls[i]:.0f}')
        if i < len(actions):
            ax.set_title(f'Engine: {engine} | Reward: {rewards[i]:.2f} | score: {np.sum(rewards[:i+1]):.2f}')
            axd[0].set_title(f'replace at k={actions[i]}' if actions[i] != 0 else f'do nothing')
            for j, (barplot, meanline) in enumerate(zip(bars, meanlines)):
                ylabels[j].set_color('g' if actions[i]==j else 'k')
                ylabels[j].set_fontweight('bold' if actions[i]==j else None)
                for k, bar in enumerate(barplot):
                    bar.set_height(dists[i,j,k])
                meanline.set_xdata(sum(zj*dists[i,j]))
        else:
            ax.set_title(f'Engine: {engine} | Final score: {np.sum(rewards):.2f}')
            axd[0].set_title(f'Terminated')
        return [pdf_plot, rul_line, rul_text, patches] + bars + meanlines + ylabels
    #                                                                 v change to -1 instead of -100
    ani = animation.FuncAnimation(fig, animate, frames=ruls.shape[0]-1, interval=100, blit=False, repeat=False)
    if savepath is None:
        vid = ani.to_html5_video()
        html = display.HTML(vid)
        display.display(html)
    else:
        writer = animation.FFMpegWriter(fps=2) 
        ani.save(f'{savepath}/{engine}-testani.mp4', writer=writer)
    plt.close()

    if return_output:
        return states, rewards

def run_test_engine(model: DQNAgent, idx:int, term:bool = True):
    model.is_test = True
    model.env.term_phase = False

    cur_data, model.env.cur_rul, _, _ = model.env.dataset.get_unit_by_id(idx)
    cur_data = cur_data.float().to(model.env.device)
    with torch.no_grad():
        pre_states = torch.cat(model.env.model(cur_data[:,model.env.dataoffset:]),dim=-1)
        model.env.states = model.env._transform_states(pre_states)
    model.env.t = 0
    model.env.terminal = 0

    state = model.env.states[0].to(model.device)
    actions, dists, cvars, rewards = [], [], [], []
    done = False
    with torch.no_grad():
        while not done:
            eval = model.dqn(state)
            action = eval.argmax(1).cpu().numpy()[0]
            actions.append(action)
            cvar = eval.detach().cpu().numpy()[0]
            cvars.append(cvar)
            dist = model.dqn.dist(state).cpu().numpy()[0]
            dists.append(dist)

            state, reward, done = model.step(action)
            rewards.append(reward)
            if state is not None:
                state = torch.FloatTensor(state).to(model.device)
            if action != 0 and term:
                #print(done)
                break
    
    actions = np.asarray(actions)
    dists = np.asarray(dists)
    cvars = np.asarray(cvars)
    rewards = np.asarray(rewards)

    try:
        final_rul = model.env.cur_rul[model.env.t].numpy()
        ruls = model.env.cur_rul[:model.env.t].numpy()
    except IndexError:
        final_rul = 0
        ruls = model.env.cur_rul.numpy()

    return final_rul, actions, rewards, dists, cvars, ruls, model.env.states.cpu().numpy(), pre_states.cpu().numpy()

def rl_engine_anim_allmodels(engineid: int, agent1: DQNAgent, agent2: DQNAgent, agent3: DQNAgent, agent4: DQNAgent, savepath: str|None = None):
    assert agent1.cvar_alpha == agent3.cvar_alpha
    assert agent2.cvar_alpha == agent2.cvar_alpha

    def act_to_string(i):
        if i == 0:
            return 'do nothing'
        elif i == 1:
            return 'replace now'
        else:
            return 'replace in 10'

    _, actions1, rewards1, dists1, cvars1, ruls1, statesCDF1, states1 = run_test_engine(agent1, engineid, term=False)
    _, actions2, rewards2, dists2, cvars2, ruls2, statesCDF2, states2 = run_test_engine(agent2, engineid, term=False)
    _, actions3, rewards3, dists3, cvars3, ruls3, statesCDF3, states3 = run_test_engine(agent3, engineid, term=False)
    _, actions4, rewards4, dists4, cvars4, ruls4, statesCDF4, states4 = run_test_engine(agent4, engineid, term=False)

    xs = np.linspace(0, ruls1[0]+20, 1000)
    #cmap = plt.get_cmap('tab10')
    cmap = ['k', 'tab:red', 'tab:blue']
    zj = agent1.support.cpu().numpy()
    delta_z = float(agent1.v_max-agent1.v_min) / (agent1.atom_size-1)

    fig = plt.figure(layout='constrained', figsize=(12,4))
    subfigs = fig.subfigures(1,6, wspace=0.02, width_ratios=[6,1,2,2,2,2])
    ax = subfigs[0].subplots()
    ax.set_title(f'Engine: {engineid} | Scores: {np.sum(rewards1[:0+1]):.2f}, {np.sum(rewards2[:0+1]):.2f}, {np.sum(rewards3[:0+1]):.2f}, {np.sum(rewards4[:0+1]):.2f}')

    mu, sig = states1[0,-1,0], states1[0,-1,1]
    wmu, wsig = states3[0,-1,0], states3[0,-1,1]
    ydata = scipy.stats.lognorm.pdf(xs, s=sig, scale=np.exp(mu))
    wydata = scipy.stats.lognorm.pdf(xs, s=wsig, scale=np.exp(wmu))
    pdf_plot, = ax.plot(xs, ydata, label='Risk-Neutral', linewidth=2)
    wpdf_plot, = ax.plot(xs, wydata, label='Risk-Averse', linewidth=2)
    ax.legend(loc='lower right', title='Forecasts')
    ax.set_ylim(0, np.max([ydata.max(),wydata.max()])*1.1)
    ax.set_ylim(0,ruls1[0]+20)
    rul_line = ax.axvline(min(128,ruls1[0]), color='k', linestyle='--')
    rul_text = ax.text(min(128,ruls1[0]), 0.99, f' RUL={ruls1[0]:.0f}' if ruls1[0]<=128 else f'RUL={ruls1[0]:.0f} \nTarget=128 ', color='k', ha='left' if ruls1[0]<=128 else 'right', va='top', transform=ax.get_xaxis_transform())
    ax.set_xlabel('RUL')

    axlabel = subfigs[1].subplots()
    axd1 = subfigs[1+1].subplots(dists1.shape[1], sharex=True)
    axd2 = subfigs[2+1].subplots(dists2.shape[1], sharex=True)
    axd3 = subfigs[3+1].subplots(dists3.shape[1], sharex=True)
    axd4 = subfigs[4+1].subplots(dists4.shape[1], sharex=True)

    subfigs[1].suptitle(f'Model:')
    subfigs[1+1].suptitle(f'Neutral-Mean')
    subfigs[2+1].suptitle(f'Neutral-CVaR')
    subfigs[3+1].suptitle(f'Averse-Mean')
    subfigs[4+1].suptitle(f'Averse-CVaR')

    axlabel.set_title('Actions')
    axlabel.axis('off')
    axd1[0].set_title(act_to_string(actions1[0]))# (f'replace at k={actions1[0]}' if actions1[0] != 0 else f'do nothing')
    axd2[0].set_title(act_to_string(actions2[0]))#(f'replace at k={actions2[0]}' if actions2[0] != 0 else f'do nothing')
    axd3[0].set_title(act_to_string(actions3[0]))#(f'replace at k={actions3[0]}' if actions3[0] != 0 else f'do nothing')
    axd4[0].set_title(act_to_string(actions4[0]))#(f'replace at k={actions4[0]}' if actions4[0] != 0 else f'do nothing')
    spinewidth = axd1[0].spines['bottom'].get_linewidth()

    bars1, bars2, bars3, bars4 = [], [], [], []
    meanlines1, meanlines2, meanlines3, meanlines4 = [], [], [], []
    ylabels= []
    for i in range(dists1.shape[1]):
        axd1[i].set_yticks([]); axd2[i].set_yticks([]); axd3[i].set_yticks([]); axd4[i].set_yticks([])#; axlabel[i].set_yticks([])
        for spine in ['bottom', 'top', 'right', 'left']:
            axd1[i].spines[spine].set_edgecolor('g' if actions1[0]==i else 'k')
            axd2[i].spines[spine].set_edgecolor('g' if actions2[0]==i else 'k')
            axd3[i].spines[spine].set_edgecolor('g' if actions3[0]==i else 'k')
            axd4[i].spines[spine].set_edgecolor('g' if actions4[0]==i else 'k')
            axd1[i].spines[spine].set_linewidth(2 if actions1[0]==i else spinewidth)
            axd2[i].spines[spine].set_linewidth(2 if actions2[0]==i else spinewidth)
            axd3[i].spines[spine].set_linewidth(2 if actions3[0]==i else spinewidth)
            axd4[i].spines[spine].set_linewidth(2 if actions4[0]==i else spinewidth)
        axd1[i].patch.set_facecolor('g' if actions1[0]==i else 'w'); axd1[i].patch.set_alpha(0.1 if actions1[0]==i else 1)
        axd2[i].patch.set_facecolor('g' if actions2[0]==i else 'w'); axd2[i].patch.set_alpha(0.1 if actions2[0]==i else 1)
        axd3[i].patch.set_facecolor('g' if actions3[0]==i else 'w'); axd3[i].patch.set_alpha(0.1 if actions3[0]==i else 1)
        axd4[i].patch.set_facecolor('g' if actions4[0]==i else 'w'); axd4[i].patch.set_alpha(0.1 if actions4[0]==i else 1)
        ylabels.append(axlabel.text(0.5,1-i/(dists1.shape[1])-0.49/(dists1.shape[1]), act_to_string(i), transform=axlabel.transAxes, ha='center', va='center', color='k'))
        c = cmap[i] #'k' if i == 0 else cmap((i-1)/10)
        bars1.append(axd1[i].bar(zj, dists1[0,i], width=delta_z, align='center', color=c, edgecolor='w'))
        bars2.append(axd2[i].bar(zj, dists2[0,i], width=delta_z, align='center', color=c, edgecolor='w'))
        bars3.append(axd3[i].bar(zj, dists3[0,i], width=delta_z, align='center', color=c, edgecolor='w'))
        bars4.append(axd4[i].bar(zj, dists4[0,i], width=delta_z, align='center', color=c, edgecolor='w'))
        meanlines1.append(axd1[i].axvline(cvars1[0,i], color='k' if i!=0 else 'r', linestyle='--'))
        meanlines2.append(axd2[i].axvline(cvars2[0,i], color='k' if i!=0 else 'r', linestyle='--'))
        meanlines3.append(axd3[i].axvline(cvars3[0,i], color='k' if i!=0 else 'r', linestyle='--'))
        meanlines4.append(axd4[i].axvline(cvars4[0,i], color='k' if i!=0 else 'r', linestyle='--'))

    def animate(i):
        #i += ruls1.shape[0]-1-20 # TEMP
        mu, sig = states1[i,-1,0], states1[i,-1,1]
        wmu, wsig = states3[i,-1,0], states3[i,-1,1]
        ydata = scipy.stats.lognorm.pdf(xs, s=sig, scale=np.exp(mu))
        wydata = scipy.stats.lognorm.pdf(xs, s=wsig, scale=np.exp(wmu))
        ax.set_ylim(0, np.max([ydata.max(),wydata.max()])*1.1)
        ax.set_xlim(0,ruls1[i]+20)
        pdf_plot.set_ydata(ydata)
        wpdf_plot.set_ydata(wydata)
        rul_line.set_xdata(min(ruls1[i],128))
        rul_text.set(position=(min(ruls1[i],128),0.99), transform=ax.get_xaxis_transform(), text=f' RUL={ruls1[i]:.0f}' if ruls1[i]<=128 else f' RUL={ruls1[i]:.0f} \nTarget=128 ', ha='left' if ruls1[i]<=128 else 'right')
        ax.set_title(f'Engine: {engineid} | Scores: {np.sum(rewards1[:i+1]):.2f}, {np.sum(rewards2[:i+1]):.2f}, {np.sum(rewards3[:i+1]):.2f}, {np.sum(rewards4[:i+1]):.2f}')

        axd1[0].set_title(act_to_string(actions1[i]))#(f'replace {actions1[i]:02d}' if actions1[i] != 0 else f'do nothing')
        axd2[0].set_title(act_to_string(actions2[i]))#(f'replace {actions2[i]:02d}' if actions2[i] != 0 else f'do nothing')
        axd3[0].set_title(act_to_string(actions3[i]))#(f'replace {actions3[i]:02d}' if actions3[i] != 0 else f'do nothing')
        axd4[0].set_title(act_to_string(actions4[i]))#(f'replace {actions4[i]:02d}' if actions4[i] != 0 else f'do nothing')
        for axd, bars, meanlines, actions, dists, cvars in zip([axd1, axd2, axd3, axd4], [bars1, bars2, bars3, bars4], [meanlines1, meanlines2, meanlines3, meanlines4],[actions1, actions2, actions3, actions4], [dists1, dists2, dists3, dists4], [cvars1, cvars2, cvars3, cvars4]):
            for j, (barplot, meanline) in enumerate(zip(bars, meanlines)):
                for spine in ['bottom', 'top', 'right', 'left']:
                    axd[j].spines[spine].set_edgecolor('g' if actions[i]==j else 'k')
                    axd[j].spines[spine].set_linewidth(2 if actions[i]==j else spinewidth)
                axd[j].patch.set_facecolor('g' if actions[i]==j else 'w')
                axd[j].patch.set_alpha(0.1 if actions[i]==j else 1)
                #ylabels[j].set_color('g' if actions[i]==j else 'k')
                #ylabels[j].set_fontweight('bold' if actions[i]==j else None)
                for k, bar in enumerate(barplot):
                    bar.set_height(dists[i,j,k])
                meanline.set_xdata(cvars[i,j])
        return [pdf_plot, rul_line, rul_text] + bars1+bars2+bars3+bars4 + meanlines1+meanlines2+meanlines3+meanlines4
    #                                                  ruls1.shape[0]-1
    ani = animation.FuncAnimation(fig, animate, frames=ruls1.shape[0]-1, interval=100, blit=False, repeat=False)
    #ani = animation.FuncAnimation(fig, animate, frames=ruls.shape[0]-1, interval=100, blit=False, repeat=False)
    if savepath is None:
        vid = ani.to_html5_video()
        html = display.HTML(vid)
        display.display(html)
    else:
        writer = animation.FFMpegWriter(fps=10)
        ani.save(f'{savepath}/{engineid}-ani.mp4', writer=writer)
    plt.close()
