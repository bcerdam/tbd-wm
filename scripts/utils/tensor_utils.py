import torch
import numpy as np
import gymnasium as gym
import ale_py
from scripts.models.agent.critic import Critic
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.sampler import sample
from dataclasses import dataclass, asdict
from typing import Tuple, List


@dataclass
class EpochTimer:
    data_init: float = 0.0
    batch_extract: float = 0.0
    ae_fwd: float = 0.0
    tokenizer: float = 0.0
    dm_fwd: float = 0.0
    loss_calc: float = 0.0
    agent_batch: float = 0.0
    agent_train: float = 0.0
    eval_episodes: float = 0.0
    plot: float = 0.0

    def reset(self):
        for field in self.__dataclass_fields__:
            setattr(self, field, 0.0)

    def total_time(self) -> float:
        return sum(asdict(self).values())
    
    def report(self, epoch_idx: int):
        print(f"--- Epoch {epoch_idx} Timing Stats ---")
        for key, value in asdict(self).items():
            print(f"{key.replace('_', ' ').title():<15}: {value:.4f}s")
        print(f"{'TOTAL':<15}: {self.total_time():.4f}s")
        print(f"----------------------------------")
            

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
        for slow_param, param in zip(ema_critic.parameters(), critic.parameters()):
            slow_param.data.copy_(slow_param.data * ema_sigma + param.data * (1 - ema_sigma))


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
    

def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per
    
