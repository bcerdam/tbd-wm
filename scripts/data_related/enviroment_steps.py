import torch
import numpy as np
import gymnasium as gym
import ale_py
from ..utils.tensor_utils import normalize_observation, reshape_observation
from gymnasium import Env
from gymnasium.wrappers import AtariPreprocessing, ClipReward
from typing import Tuple, List, Dict
from scripts.models.agent.actor import Actor
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.sampler import sample
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from torch.distributions import OneHotCategorical


def gather_steps(env:Env, 
                 observation:np.ndarray, 
                 state:Dict, 
                 features:torch.Tensor, 
                 episode_start:bool, 
                 lives:int,
                 env_steps_per_epoch: int,
                 actor:Actor, 
                 encoder:CategoricalEncoder, 
                 tokenizer:Tokenizer, 
                 xlstm_dm:XLSTM_DM, 
                 latent_dim:int, 
                 codes_per_latent:int, 
                 device:str) -> Tuple[List[np.ndarray], List[np.int64], List[np.float64], List[bool], List[bool]]:

    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminations = [] 
    all_episode_starts = []

    with torch.no_grad():
        for step in range(env_steps_per_epoch):
            all_episode_starts.append(episode_start)
            all_observations.append(observation)

            observation_tensor = torch.from_numpy(observation).unsqueeze(0).unsqueeze(0).to(device=device)
            latent = encoder.forward(observations_batch=observation_tensor, 
                                    batch_size=1, 
                                    sequence_length=1, 
                                    latent_dim=latent_dim, 
                                    codes_per_latent=codes_per_latent)
            sampled_latent = sample(latents_batch=latent, batch_size=1, sequence_length=1)

            action_array = np.zeros(env.action_space.n, dtype=np.float32)
            env_state = torch.concat([sampled_latent.view(1, 1, -1), features], dim=-1)
            action_logits = actor(state=env_state)
            policy = OneHotCategorical(logits=action_logits)
            action = torch.argmax(policy.sample()).item()
            action_array[action] = 1.0
            tensor_action_array = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device)
            all_actions.append(action_array)

            token = tokenizer.forward(latents_sampled_batch=sampled_latent, actions_batch=tensor_action_array)
            _, _, _, features, state = xlstm_dm.step(tokens_batch=token, state=state)

            episode_start = False
            observation, reward, termination, truncated, info = env.step(action)
            current_lives = info.get("lives", 0)
            life_loss = current_lives < lives
            lives = current_lives
            observation = reshape_observation(normalize_observation(observation=observation))

            all_rewards.append(reward)
            all_terminations.append(termination or life_loss)

            if termination or truncated:
                observation, info = env.reset()
                lives = info.get("lives", 0)
                observation = reshape_observation(normalize_observation(observation=observation))
                episode_start = True
                state = None
                features = torch.zeros(1, 1, tokenizer.embedding_dim, device=device)

    all_observations = np.array(all_observations)
    all_actions = np.array(all_actions)
    all_rewards = np.array(all_rewards)
    all_terminations = np.array(all_terminations)
    all_episode_starts = np.array(all_episode_starts)

    return all_observations, all_actions, all_rewards, all_terminations, all_episode_starts, observation, state, features, episode_start, lives