import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset


class AtariDataset(Dataset):
    def __init__(self, sequence_length:int, total_env_steps:int, env_actions:int, device:str, dtype) -> None:

        self.sequence_length = sequence_length
        self.total_env_steps = total_env_steps
        self.device = device
        self.dtype = dtype

        self.observations = np.zeros(shape=(self.total_env_steps, 3, 64, 64))
        self.actions = np.zeros(shape=(self.total_env_steps, env_actions))
        self.rewards = np.zeros(shape=(self.total_env_steps))
        self.terminations = np.zeros(shape=(self.total_env_steps))

        self.pointer = 0


    def update(self, observation:np.ndarray, 
                     action:np.ndarray, 
                     reward:int, 
                     termination:bool) -> None:
        
        self.observations[self.pointer, :, :, :] = observation
        self.actions[self.pointer, :] = action
        self.rewards[self.pointer] = reward
        self.terminations[self.pointer] = termination

        self.pointer += 1

    
    def extract_random_batch(self, batch_size:int):
        current_length = self.pointer-self.sequence_length
        random_idxs = np.random.randint(0, current_length, batch_size)
        seq_indxs = random_idxs[:, None] + np.arange(self.sequence_length)

        observations_batch = torch.from_numpy(self.observations[seq_indxs]).to(device=self.device).to(dtype=self.dtype)
        actions_batch = torch.from_numpy(self.actions[seq_indxs]).to(device=self.device)
        rewards_batch = torch.from_numpy(self.rewards[seq_indxs]).to(device=self.device)
        terminations_batch = torch.from_numpy(self.terminations[seq_indxs]).to(device=self.device)

        return observations_batch, actions_batch, rewards_batch, terminations_batch

        
    def __len__(self) -> int:
        return self.pointer-self.sequence_length
    

    def __getitem__(self, index:int) -> Tuple:
        return (self.observations[index:index+self.sequence_length], 
                self.actions[index:index+self.sequence_length], 
                self.rewards[index:index+self.sequence_length], 
                self.terminations[index:index+self.sequence_length])