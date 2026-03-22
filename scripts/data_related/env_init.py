import torch
import numpy as np
import gymnasium as gym
import ale_py
from typing import Tuple
from gymnasium.wrappers import AtariPreprocessing, ClipReward
from ..models.categorical_vae.encoder import CategoricalEncoder
from ..models.categorical_vae.sampler import sample
from ..utils.tensor_utils import normalize_observation, reshape_observation, FireOnLifeLossWrapper
from ..models.dynamics_modeling.transformer_model import StochasticTransformerKVCache
from ..models.dynamics_modeling.attention_blocks import get_subsequent_mask_with_batch_length


def env_init(env_name:str, 
             noop_max:int, 
             frame_skip:int, 
             screen_size:int, 
             terminal_on_life_loss:bool, 
             min_reward:float, 
             max_reward:float, 
             encoder:CategoricalEncoder, 
             latent_dim:int, 
             codes_per_latent:int, 
             device:str,
             storm_transformer:StochasticTransformerKVCache) -> Tuple:
    
    gym.register_envs(ale_py)
    env = gym.make(id=env_name, frameskip=1, full_action_space=False)
    env = FireOnLifeLossWrapper(env)
    env = AtariPreprocessing(env=env, 
                            noop_max=noop_max, 
                            frame_skip=frame_skip, 
                            screen_size=screen_size, 
                            terminal_on_life_loss=terminal_on_life_loss, 
                            grayscale_obs=False)
    env = ClipReward(env=env, min_reward=min_reward, max_reward=max_reward)

    observation, info = env.reset()

    random_action_idx = env.action_space.sample()
    action_array = np.zeros(env.action_space.n, dtype=np.float32)
    action_array[random_action_idx] = 1.0
    tensor_action = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device) # a_(t)

    observation = reshape_observation(normalize_observation(observation=observation))
    observation_tensor = torch.from_numpy(observation).unsqueeze(0).unsqueeze(0).to(device=device) # o_t
    latent_t = encoder.forward(observations_batch=observation_tensor,
                                batch_size=1,
                                sequence_length=1,
                                latent_dim=latent_dim,
                                codes_per_latent=codes_per_latent)
    latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) # z_t

    flattened_sample = latent_t.flatten(start_dim=2)

    dist_feat = storm_transformer.forward_with_kv_cache(samples=flattened_sample, action=random_action_idx)

    lives = info.get("lives", 0)
    return env, observation, random_action_idx, lives, dist_feat