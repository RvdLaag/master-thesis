import sys
import os
# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.forecast_model import TruncNormNetwork


class ForecastEnv:
    def __init__(
        self,
        n_actions: int,
        planning_window: int,
        n_observations: int,
        model: str | TruncNormNetwork,
        dataset: Dataset,
        engines: np.ndarray,
        test_engines: np.ndarray,
        window_size: int,
        seed: int,
        cf: float = 4, # 2
        c1: float = 0.1, # 0.01,
        c0: float = 2, # 1
        cn: float = 0.1 # changed from zero | 0.01
    ) -> None:
        
        self.n_actions = n_actions
        self.planning_window = planning_window
        self.n_observations = n_observations
        if isinstance(model, str):
            self.model = torch.load(model)
        elif isinstance(model, TruncNormNetwork): # TODO: accept other network types / any nn.Module ?
            self.model = model
        else:
            raise TypeError("Model must be string path to file or TruncNormNetwork")
        self.model.eval()
        self.dataset = dataset
        self.engines = engines
        self.test_engines = test_engines
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.terminal = 0 # None
        self.cf, self.c1, self.c0, self.cn = cf, c1, c0, cn
        #self.cb = self.planning_window + np.sqrt((1+c0)*2*self.planning_window**2/cn)

        self.current_engine = self.rng.choice(engines)
        
        cur_data, self.cur_rul, _, _ = self.dataset.get_unit_by_id(self.current_engine) # tensors
        cur_data = cur_data.float().to(self.device)
        self.dataoffset = int(cur_data.shape[1] - window_size)
        with torch.no_grad():
            self.states = self._transform_states(torch.cat(self.model(cur_data[:,self.dataoffset:]), dim=-1))
        self.t = 0

        self.term_phase = False

        print(self.c0 - self.c1*self.planning_window)
        print(f'{self.c0=}, {self.c1=}, {self.cf=}, {self.cn=}')

    def sample_action(self) -> int:
        """Select random action""" #  V -self.n_actions or 0
        a = self.rng.integers(low=0, high=self.n_actions) # change low and clamp >= 0 to give more prio to action=0
        return min(max(a, 0),self.n_actions-1)

        
    def step(self, action):
        """Take a step forward"""
        reward = self._get_reward(action)
        self.t += 1
        
        if self.t >= len(self.cur_rul):
            return None, reward, True
        
        next_state = self.states[self.t].cpu().numpy()
        if action == 0:
            return next_state, reward, False #reward==-self.cf and self.term_phase # False
        else:
            return next_state, reward, self.term_phase #False#False # False or True ?
        #return np.concatenate((next_state, [self.terminal])), reward, False
    
    def _get_reward(self, action: int, rul: int|None = None) -> float: # mainrun
        if rul is None:
            rul = self.cur_rul[self.t].numpy()

        reward = 0.
        if action == 0: # we do nothing
            if rul <= self.planning_window:
                reward -= self.cf # failed
            else:
                reward += self.cn
        elif action == 1: # emergency replacement
            reward -= self.c0
        else: # replace inside planning_window
            if rul < self.planning_window: # < or <= ????
                reward -= self.cf
            else:
                reward -= self.c0 - self.c1*self.planning_window
        return reward
    
    def _get_reward_mainrun2(self, action: int, rul: int|None = None) -> float:
        if rul is None:
            rul = self.cur_rul[self.t].numpy()

        
        if action == 0: # we do nothing
            if rul <= self.planning_window:
                return -self.cf # failed
            else:
                return self.cn
        elif action == 1: # emergency replacement
            return -self.c0
        else: # replace inside planning_window
            if rul <= self.planning_window: # < or <= ????
                return -self.cf
            else:
                #print(max(self.cb-rul,0)**2 *self.cn/self.planning_window**2 - self.c0)
                return (max(self.cb-rul,0)**2 *self.cn/(2*self.planning_window**2) - self.c0)
            
    def _get_reward_fucked(self, action: int, rul: int|None = None) -> float:
        if rul is None:
            rul = self.cur_rul[self.t].numpy()

        
        if action == 0: # we do nothing
            if rul <= self.planning_window:
                return -self.cf # failed
            else:
                return self.cn
        elif action == 1: # emergency replacement
            return -self.c0
        else: # replace inside planning_window
            if rul <= self.planning_window: # < or <= ????
                return -self.cf
            else:
                #print(max(self.cb-rul,0)**2 *self.cn/self.planning_window**2 - self.c0)
                return self.c1*self.planning_window - self.c0

    
    def step_old2(self, action):
        """Take a step forward"""
        if self.terminal != 0: # is not None
            self.terminal -= 1
        
        reward = self._get_reward(action)

        self.t += 1
        if self.t >= len(self.cur_rul) or self.terminal == 0: # 0 or 1?
            self.terminal = 0 # None # might not be needed | done in self.reset()
            if self.terminal == 1 and False:
                reward -= self.cur_rul[self.t]/10 # add reward based on rul
            return None, reward, True 
        with torch.no_grad():
            next_state = self.states[self.t].cpu().numpy()
        #return next_state, reward, False
        return np.concatenate((next_state, [self.terminal])), reward, False
        

    def _get_reward_old2(self, action: int, rul: int|None = None) -> float:
        if rul is None:
            rul = self.cur_rul[self.t].numpy()
        reward = 0.
        if self.terminal == 0 : # is None
            if action == 0:
                if rul < self.n_actions:
                    reward -= self.cf # failed to schedule before failure
                else:
                    reward += self.cn # correctly didnt schedule
            else: # action = replace at k
                self.terminal = action
                if rul <= action:
                    reward -= self.cf # replace too late
                else:
                    reward -= self.c0 - self.c1*action
        else: # terminal = t
            if action == 0:
                if rul < self.terminal: # self.n_actions:
                    reward -= self.cf
                else:
                    reward += self.cn
                self.terminal = 0 # None # stop replacement
            else: # action = replace at k
                if rul <= action:
                    reward -= self.cf
                else:
                    if action == self.terminal:
                        reward += self.cn # correctly stuck to replacement scheduled
                    else:
                        reward -= self.c1*(self.n_actions-action) # do not incur c0 again
                self.terminal = action
        return reward
    
    def reset(self):
        self.current_engine = self.rng.choice(np.setdiff1d(self.engines, self.current_engine))
        cur_data, self.cur_rul, _, _ = self.dataset.get_unit_by_id(self.current_engine) # tensors
        cur_data = cur_data.float().to(self.device)
        with torch.no_grad():
            self.states = self._transform_states(torch.cat(self.model(cur_data[:,self.dataoffset:]), dim=-1))
        self.t = 0
        #self.t = self.rng.integers(low=0, high=len(self.cur_rul)-2*self.planning_window)
        #self.t = int(self.rng.triangular(0, len(self.cur_rul)-2*self.planning_window, len(self.cur_rul)-2*self.planning_window))
        self.terminal = 0 # None
        #return np.concatenate((self.states[0].cpu().numpy(), [self.terminal]))
        return self.states[0].cpu().numpy() # self.states[self.t].cpu().numpy()
    
    def test_states(self, idx: int):
        assert idx>=0 and idx<len(self.test_engines)
        data, ruls, _, _ = self.dataset.get_unit_by_id(self.test_engines[idx])
        data = data.float().to(self.device)
        with torch.no_grad():
            states = self._transform_states(torch.cat(self.model(data[:,self.dataoffset:]), dim=-1))
        return states, ruls.numpy()
    
    def set_test_states(self, idx: int):
        assert idx>=0 and idx<len(self.test_engines)
        data, ruls, _, _ = self.dataset.get_unit_by_id(self.test_engines[idx])
        data = data.float().to(self.device)
        with torch.no_grad():
            pre_states = torch.cat(self.model(data[:,self.dataoffset:]), dim=-1)
            states = self._transform_states(pre_states)
        self.cur_rul = ruls
        self.states = states
        self.t = 0
        self.terminal = 0
        return pre_states, states, ruls.numpy()
    
    def test_rewards(self, actions, ruls) -> Tuple[np.ndarray, np.ndarray]:
        rewards = []
        terminated = []
        self.terminal = None
        for action, rul in zip(actions, ruls):
            if self.terminal is not None:
                self.terminal -= 1
            if self.terminal == 0 or (terminated[-1] if terminated else False):
                terminated.append(True)
                self.terminal = None
            else:
                terminated.append(False)
            rewards.append(self._get_reward(action, rul))

        return np.asarray(rewards), np.asarray(terminated)
    


    
    def test_rewards_old(self, actions, ruls) -> np.ndarray:
        rewards = []
        for a, r in zip(actions, ruls):
            if a == 0:
                if r <= self.n_actions-1:
                    rewards.append(-self.cf)
                else:
                    rewards.append(self.cn)
            else:
                if r >= a:
                    rewards.append(-(self.c0-self.c1*a))
                else:
                    rewards.append(-self.cf)
        return np.asarray(rewards)

    def _get_reward_2old(self, action: int, rul: int|None = None) -> float:
        if rul is None:
            rul = self.cur_rul[self.t].numpy()
        reward = 0.
        if action == 0:
            if self.terminal is not None:
                self.terminal = None # cancel replacement
                reward += self.c0 - self.c1*(self.n_actions-1) # refund cost of replacement, but not of scheduling replacement
            if rul < self.n_actions:
                reward -= self.cf # failed to schedule when RUL <= D
            else:
                reward += self.cn # correctly didn't schedule
        else: # action not zero
            if self.terminal is None:
                self.terminal = action # terminate in -action- cycles to replace
                if rul >= action:
                    reward -= self.c0 - self.c1*action # replacement in time
                else:
                    reward -= self.cf # replacement is too late
            elif self.terminal > action:
                self.terminal = action # reschedule replacement incur extra cost
                reward -= self.c1*((self.n_actions-1)-action) # dont incur c0 again just c1*action
            elif self.terminal < action:
                pass # don't do anything | alternative is to update terminal ?
            else: # self.terminal == action:
                reward += self.cn # positive reward
        return reward

    def _get_reward_old(self, action) -> float:
        if action == 0:
            if self.cur_rul[self.t].numpy() <= self.n_actions-1:
                return -self.cf
            else:
                return self.cn
        else:
            if self.cur_rul[self.t].numpy() >= action:
                return -(self.c0-self.c1*action)
            else:
                return -self.cf

    def _transform_states(self, states):
        # transform states to RL input
        # return states # indentity
        out = torch.zeros((states.shape[0],self.n_observations[0]), dtype=states.dtype, device=states.device)
        for i in range(1,self.n_observations[0]+1):
            out[:,i-1] = (1+torch.special.erf((torch.log(torch.tensor(i))-states[:,-1,0])/(states[:,-1,1]*torch.sqrt(torch.tensor(2)))))/2
        return out