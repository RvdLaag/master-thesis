import sys
import os
# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Dict, List, Callable

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
from IPython import display
import scipy.stats

from src.utils.environment import ForecastEnv
from src.utils.replaybuffer import ReplayBuffer
from src.models.rl_network import Network



class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env: 
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
    """

    def __init__(
        self, 
        env: ForecastEnv,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_func: Callable,
        seed: int,
        lr: float = 0.001,
        max_epsilon: float = 0.9,
        min_epsilon: float = 0.05,
        gamma: float = 0.9,#9, #0.99 BEFORE
        # Categorical DQN parameters
        v_min: float = -10.0,
        v_max: float = 0.0,
        atom_size: int = 10,
        cvar_alpha: float = 1.0, # alpha = 1 -> mean action
        load: str|None = None,
    ):
        obs_dim = env.n_observations# + 1
        action_dim = env.n_actions 
        
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_func = epsilon_func
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)
        self.cvar_alpha = cvar_alpha

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, atom_size, self.support, self.cvar_alpha
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, atom_size, self.support, self.cvar_alpha
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        if isinstance(load, str):
            loaddict = torch.load(load, weights_only=True)
            self.dqn.load_state_dict(loaddict['dqn'])
            self.dqn_target.load_state_dict(loaddict['dqn_target'])
            self.optimizer.load_state_dict(loaddict['optimizer'])

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def save(self, path: str):
        # TODO save ReplayBuffer? pickle?
        torch.save({'dqn': self.dqn.state_dict(),
                    'dqn_target': self.dqn_target.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, path)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = self.env.sample_action()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device),
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done = self.env.step(action)
        
        if not self.is_test and next_state is not None:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)
        if torch.isnan(loss):
            print(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def test(self, idx):
        self.is_test = True
        pre_states, states, ruls = self.env.set_test_states(idx)
        #state = torch.cat((states[0], torch.tensor([0], device='cuda')))
        state = states[0].to(self.device)
        
        actions = []
        dists = []
        cvars = []
        rewards = []

        done = False
        with torch.no_grad():
            while not done:
                eval = self.dqn(state)

                action = eval.argmax(1).cpu().numpy()[0]
                actions.append(action)
                cvar = eval.detach().cpu().numpy()[0]
                cvars.append(cvar)
                dist = self.dqn.dist(state).cpu().numpy()[0]
                dists.append(dist)

                state, reward, done = self.step(action)
                rewards.append(reward)
                if state is not None:
                    state = torch.FloatTensor(state).to(self.device)
        
        actions = np.asarray(actions)
        dists = np.asarray(dists)
        cvars = np.asarray(cvars)
        rewards = np.asarray(rewards)
        return pre_states.cpu().numpy(), states.cpu().numpy(), ruls, rewards, actions, dists, cvars


        
    def train(self, num_frames: int, to_plot: bool = False, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                #if frame_idx == 12_000:
                    #self.env.term_phase = True
                self.epsilon = self.epsilon_func(update_cnt, self.min_epsilon, self.max_epsilon)
                """
                if self.epsilon_phase == 'decrease':
                    # linearly decrease epsilon 
                    self.epsilon = max(
                        self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                        ) * self.epsilon_decay
                    )
                else:
                    # linearly increase epsilon
                    self.epsilon = min(
                        self.max_epsilon*.25, self.epsilon + (
                            self.max_epsilon - self.min_epsilon
                        ) * self.epsilon_decay
                    )
                """
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

                # swap epsilon decay phase
                #if update_cnt % int(1.5/self.epsilon_decay) == 0 and update_cnt < 20000: # TODO
                    #print('swapped')
                    #self.epsilon_phase = 'decrease' if self.epsilon_phase == 'increase' else 'increase'

            # plotting
            if to_plot and frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn_target(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])

        loss = -(proj_dist * log_p).sum(1).mean()

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()