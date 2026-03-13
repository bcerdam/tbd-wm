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


    def update(self, observations: np.ndarray, 
                     actions: np.ndarray, 
                     rewards: np.ndarray, 
                     terminations: np.ndarray) -> None:
        
        if self.observations is None:
            self.observations = observations
            self.actions = actions
            self.rewards = rewards
            self.terminations = terminations
        else:
            self.observations = np.concatenate([self.observations, observations], axis=0)
            self.actions = np.concatenate([self.actions, actions], axis=0)
            self.rewards = np.concatenate([self.rewards, rewards], axis=0)
            self.terminations = np.concatenate([self.terminations, terminations], axis=0)


    def __len__(self) -> int:
        return len(self.observations)-self.sequence_length
    

    def __getitem__(self, index:int) -> Tuple:
        return (self.observations[index:index+self.sequence_length], 
                self.actions[index:index+self.sequence_length], 
                self.rewards[index:index+self.sequence_length], 
                self.terminations[index:index+self.sequence_length])