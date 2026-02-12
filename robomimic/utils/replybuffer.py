import numpy as np
import torch
from copy import deepcopy

from collections import OrderedDict

# TODO a meta class for a trajectory
class TransitionBuffer():
    def __init__(self):
        self.data = []
    
    def add_step(self, obs, next_obs, action, base_action, reward, done):
        self.data.append({
            "obs": deepcopy(obs),
            "next_obs": deepcopy(next_obs),
            "actions": action,
            "base_actions": base_action,
            "rewards": reward,
            "dones": done
        })
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, capacity, obs_keys):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs_keys = obs_keys
        self.keys = ["obs", "next_obs", "actions", "base_actions", "rewards", "dones"]
        
        self.buffers = {
            "obs": {k: [] for k in obs_keys},
            "next_obs": {k: [] for k in obs_keys},
            "actions": [],
            "base_actions": [], 
            "rewards": [],
            "dones": []
        }

    def add(self, transition_buffer:TransitionBuffer):
        for transition in transition_buffer:
            if self.size < self.capacity:
                for k in self.obs_keys:
                    self.buffers["obs"][k].append(transition["obs"][k])
                    self.buffers["next_obs"][k].append(transition["next_obs"][k])
                self.buffers["actions"].append(transition["actions"])
                self.buffers["base_actions"].append(transition["base_actions"])
                self.buffers["rewards"].append(transition["rewards"])
                self.buffers["dones"].append(transition["dones"])
                self.size += 1
            else:
                idx = self.ptr
                for k in self.obs_keys:
                    self.buffers["obs"][k][idx] = transition["obs"][k]
                    self.buffers["next_obs"][k][idx] = transition["next_obs"][k]
                self.buffers["actions"][idx] = transition["actions"]
                self.buffers["base_actions"][idx] = transition["base_actions"]
                self.buffers["rewards"][idx] = transition["rewards"]
                self.buffers["dones"][idx] = transition["dones"]
                self.ptr = (self.ptr + 1) % self.capacity

    def _pop_oldest(self):
        for k in self.obs_keys:
            self.buffers["obs"][k].pop(0)
            self.buffers["next_obs"][k].pop(0)
        self.buffers["actions"].pop(0)
        self.buffers["base_actions"].pop(0)
        self.buffers["rewards"].pop(0)
        self.buffers["dones"].pop(0)
        self.size -= 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item = {
            "obs": {k: torch.tensor(self.buffers["obs"][k][idx], dtype=torch.float32) for k in self.obs_keys},
            "next_obs": {k: torch.tensor(self.buffers["next_obs"][k][idx], dtype=torch.float32) for k in self.obs_keys},
            "actions": torch.tensor(self.buffers["actions"][idx], dtype=torch.float32),
            "base_actions": torch.tensor(self.buffers["base_actions"][idx], dtype=torch.float32),
            "rewards": torch.tensor(self.buffers["rewards"][idx], dtype=torch.float32),
            "dones": torch.tensor(self.buffers["dones"][idx], dtype=torch.float32)
        }
        return item