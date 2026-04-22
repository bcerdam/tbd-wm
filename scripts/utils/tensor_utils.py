import torch
import numpy as np
import gymnasium as gym
import ale_py
from scripts.models.agent.critic import Critic
from collections import deque


class MaxLast2FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
 
    def step(self, action):
        total_reward = 0
        obs_buffer = deque(maxlen=2)
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        if len(obs_buffer) == 1:
            obs = obs_buffer[0]
        else:
            obs = np.max(np.stack(obs_buffer), axis=0)
        return obs, total_reward, done, truncated, info
    

class LifeLossInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives_info = None
 
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_lives_info = info["lives"]
        if current_lives_info < self.lives_info:
            info["life_loss"] = True
            self.lives_info = info["lives"]
        else:
            info["life_loss"] = False
        return observation, reward, terminated, truncated, info
 
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.lives_info = info["lives"]
        info["life_loss"] = False
        return observation, info


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