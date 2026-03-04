import os
import h5py
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset


class AtariDataset(Dataset):
    def __init__(self, sequence_length:int) -> None:

        self.sequence_length = sequence_length

        self.observations = None
        self.actions = None
        self.rewards = None
        self.terminations = None
        self.episode_starts = None


    def update(self, observations: np.ndarray, 
                     actions: np.ndarray, 
                     rewards: np.ndarray, 
                     terminations: np.ndarray, 
                     episode_starts: np.ndarray) -> None:
        
        if self.observations is None:
            self.observations = observations
            self.actions = actions
            self.rewards = rewards
            self.terminations = terminations
            self.episode_starts = episode_starts
        else:
            self.observations = np.concatenate([self.observations, observations], axis=0)
            self.actions = np.concatenate([self.actions, actions], axis=0)
            self.rewards = np.concatenate([self.rewards, rewards], axis=0)
            self.terminations = np.concatenate([self.terminations, terminations], axis=0)
            self.episode_starts = np.concatenate([self.episode_starts, episode_starts], axis=0)


    def __len__(self):
        return len(self.observations)-self.sequence_length
    

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.observations[index:index+self.sequence_length], 
                self.actions[index:index+self.sequence_length], 
                self.rewards[index:index+self.sequence_length], 
                self.terminations[index:index+self.sequence_length])