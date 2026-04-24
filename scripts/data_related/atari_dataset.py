import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset


class AtariDataset(Dataset):
    def __init__(self, sequence_length:int, total_env_steps:int, env_actions:int, device:str, dtype, temperature:float=20.0) -> None:

        self.sequence_length = sequence_length
        self.total_env_steps = total_env_steps
        self.env_actions = env_actions
        self.device = device
        self.dtype = dtype

        # --- NEW: Prioritization parameters ---
        self.temperature = temperature
        self.sample_visits = torch.zeros(size=(self.total_env_steps,), dtype=torch.long, device=device)
        # --------------------------------------

        self.observations = torch.zeros(size=(self.total_env_steps, 3, 64, 64), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(size=(self.total_env_steps,), dtype=torch.uint8, device=device)
        self.rewards = torch.zeros(size=(self.total_env_steps,), dtype=torch.float, device=device)
        self.terminations = torch.zeros(size=(self.total_env_steps,), dtype=torch.bool, device=device)

        self.pointer = 0


    def update(self, observation:np.ndarray, 
                     action:np.ndarray, 
                     reward:int, 
                     termination:bool) -> None:
        
        self.observations[self.pointer, :, :, :] = torch.from_numpy(observation)
        self.actions[self.pointer] = int(action)
        self.rewards[self.pointer] = reward
        self.terminations[self.pointer] = bool(termination)

        self.pointer += 1

    
    def extract_random_batch(self, batch_size:int, for_world_model:bool):
        current_length = self.pointer-self.sequence_length

        if for_world_model == True:
            visits = self.sample_visits[:current_length].float()
            probs = F.softmax(visits / -self.temperature, dim=0)
            random_idxs = torch.multinomial(probs, batch_size, replacement=True).unsqueeze(1)
            unique_idxs, counts = torch.unique(random_idxs, return_counts=True)
            self.sample_visits[unique_idxs] += counts.to(self.sample_visits.dtype)
        else:
            random_idxs = torch.randint(0, current_length, (batch_size, 1), device=self.device)

        seq_indxs = random_idxs + torch.arange(self.sequence_length, device=self.device)

        observations_batch = self.observations[seq_indxs].to(torch.float16)/255.0

        raw_actions = self.actions[seq_indxs].long()
        actions_batch = F.one_hot(raw_actions, num_classes=self.env_actions).to(torch.uint8)

        rewards_batch = self.rewards[seq_indxs]
        terminations_batch = self.terminations[seq_indxs]

        return observations_batch, actions_batch, rewards_batch, terminations_batch

        
    def __len__(self) -> int:
        return self.pointer-self.sequence_length
    

    def __getitem__(self, index:int) -> Tuple:
        return (self.observations[index:index+self.sequence_length], 
                self.actions[index:index+self.sequence_length], 
                self.rewards[index:index+self.sequence_length], 
                self.terminations[index:index+self.sequence_length])