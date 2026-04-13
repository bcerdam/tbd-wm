import torch
import numpy as np
import gymnasium as gym
import ale_py
from scripts.models.agent.critic import Critic


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar
    
        
def normalize_observation(observation:np.ndarray) -> np.ndarray:
    normalized_observation = observation.astype(np.float32)/255.0
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
        for slow_param, param in zip(ema_critic.parameters(), critic.parameters()):
            slow_param.data.copy_(slow_param.data * ema_sigma + param.data * (1 - ema_sigma))

    
def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per