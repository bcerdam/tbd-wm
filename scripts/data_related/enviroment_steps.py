import torch
import numpy as np
from ..utils.tensor_utils import normalize_observation, reshape_observation
from gymnasium import Env
from typing import Tuple, List, Dict
from scripts.models.agent.actor import Actor
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.sampler import sample
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from torch.distributions import OneHotCategorical
import gymnasium as gym
import ale_py
from typing import Tuple
from gymnasium.wrappers import AtariPreprocessing, ClipReward
from ..utils.tensor_utils import normalize_observation, reshape_observation


def gather_steps(env:Env, 
                 observation:np.ndarray, 
                 action:int, 
                 lives:int,
                 features:torch.Tensor,
                 state:dict, 
                 env_steps_per_epoch: int,
                 actor:Actor, 
                 encoder:CategoricalEncoder, 
                 tokenizer:Tokenizer, 
                 xlstm_dm:XLSTM_DM, 
                 latent_dim:int, 
                 codes_per_latent:int, 
                 device:str, 
                 context_length:int, 
                 embedding_dim:int) -> Tuple:
    
    all_observations, all_actions, all_rewards, all_terminations = [], [], [], []

    # _, _, _, features = xlstm_dm.forward(tokens_batch=context_tokens)
    # _, _, _, features, state = xlstm_dm.step(tokens_batch=context_tokens, state=state) # Maybe it only needs 1 token, instead of batch context tokens
    # features = features[:, -1:, :] # h_t -> (token_t)

    action_array = np.zeros(env.action_space.n, dtype=np.float32)
    action_array[action] = 1.0

    # random action: a_t
    # token = (z_t, a_t)
    # forward = h_t
    # step: o_t+1, r_t+1, t_t+1 = env.step(a_t)
    # state = (z_(t+1), h_t)
    # action: a_t+1
    # token (z_t+1, a_t+1)
    # forward: h_t+1
    # step: 0_t+2, r_t+2, t_t+2 = env.step(a_t+1)
    # state = (z_t+2, h_t+1)

    with torch.no_grad():
        for step in range(env_steps_per_epoch):
            next_observation, next_reward, next_termination, next_truncated, info = env.step(action) # o_(t+2), r_(t+2), t_(t+2), a_(t+1)

            all_observations.append(observation) # o_t+1
            all_actions.append(action_array) # a_t+1
            all_rewards.append(next_reward) # r_(t+2)

            current_lives = info.get("lives", 0)
            life_loss = current_lives < lives
            lives = current_lives
            done = next_termination or life_loss

            all_terminations.append(done) # t_(t+2)

            next_observation = reshape_observation(normalize_observation(observation=next_observation))
            observation_tensor = torch.from_numpy(next_observation).unsqueeze(0).unsqueeze(0).to(device=device) # o_t+2
            latent_t = encoder.forward(observations_batch=observation_tensor,
                                        batch_size=1,
                                        sequence_length=1,
                                        latent_dim=latent_dim,
                                        codes_per_latent=codes_per_latent)
            latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) # z_t+2

            env_state_vec = torch.cat([latent_t.view(1, 1, -1), features], dim=-1) # (z_t+2, h_t+1)

            action_logits = actor(state=env_state_vec)
            action_idx = torch.argmax(OneHotCategorical(logits=action_logits).sample()).item()
            action_array = np.zeros(env.action_space.n, dtype=np.float32)
            action_array[action_idx] = 1.0
            tensor_action = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device) # a_(t+2)

            observation = next_observation
            action = action_idx

            token = tokenizer.forward(latents_sampled_batch=latent_t, actions_batch=tensor_action) # token_t -> (z_t+2, a_t+2)

            # context_tokens = torch.cat([context_tokens, token], dim=1)[:, -context_length:]
            # _, _, _, features = xlstm_dm.forward(tokens_batch=context_tokens)
            _, _, _, features, state = xlstm_dm.step(tokens_batch=token, state=state) # Maybe it only needs 1 token, instead of batch context tokens
            features = features[:, -1:, :]

            if next_termination or next_truncated:
                observation, info = env.reset()
                observation = reshape_observation(normalize_observation(observation=observation)) # o_t

                observation_tensor = torch.from_numpy(observation).unsqueeze(0).unsqueeze(0).to(device=device) # o_t+2
                latent_t = encoder.forward(observations_batch=observation_tensor,
                                            batch_size=1,
                                            sequence_length=1,
                                            latent_dim=latent_dim,
                                            codes_per_latent=codes_per_latent)
                latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) # z_t+2

                action = env.action_space.sample() # a_t
                action_array = np.zeros(env.action_space.n, dtype=np.float32)
                action_array[action] = 1.0
                tensor_action = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device) # a_(t)

                token = tokenizer.forward(latents_sampled_batch=latent_t, actions_batch=tensor_action) # token_t -> (z_t+2, a_t+2)
                # context_tokens = token
                state = {}

                # _, _, _, features = xlstm_dm.forward(tokens_batch=context_tokens)
                _, _, _, features, state = xlstm_dm.step(tokens_batch=token, state=state) # Maybe it only needs 1 token, instead of batch context tokens


                features = features[:, -1:, :]

                lives = info.get("lives", 0)

                                
    last_observation = observation
    observations = np.array(all_observations)
    actions = np.array(all_actions)
    rewards = np.array(all_rewards)
    terminations = np.array(all_terminations)

    return observations, actions, rewards, terminations, last_observation, action, lives, features, state