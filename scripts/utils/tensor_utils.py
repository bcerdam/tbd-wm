import torch
import numpy as np
import gymnasium as gym
import ale_py
from ..data_related.atari_dataset import AtariDataset
from torch.utils.data import default_collate
from scripts.models.agent.critic import Critic


def normalize_observation(observation:np.ndarray) -> np.ndarray:
    normalized_observation = (observation.astype(np.float32)/127.5)-1.0
    return normalized_observation


def reshape_observation(observation:np.ndarray) -> np.ndarray:
    reshaped_observation = np.moveaxis(observation, -1, 0)
    return reshaped_observation

def env_n_actions(env_name:str) -> int:
    gym.register_envs(ale_py)
    env = gym.make(id=env_name)
    n_actions = env.action_space.n
    return n_actions


def update_ema_critic(ema_sigma:float, critic:Critic, ema_critic:Critic) -> None:
    with torch.no_grad():
        for slow_param, fast_param in zip(ema_critic.parameters(), critic.parameters()):
            slow_param.data.mul_(ema_sigma).add_(fast_param.data, alpha=(1 - ema_sigma))