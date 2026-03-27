# import torch
# import numpy as np
# from ..utils.tensor_utils import normalize_observation, reshape_observation
# from gymnasium import Env
# from typing import Tuple, List, Dict
# from scripts.models.agent.actor import Actor
# from scripts.models.categorical_vae.encoder import CategoricalEncoder
# from scripts.models.categorical_vae.sampler import sample
# from scripts.models.dynamics_modeling.tokenizer import Tokenizer
# from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
# from torch.distributions import OneHotCategorical
# import gymnasium as gym
# import ale_py
# from typing import Tuple
# from gymnasium.wrappers import AtariPreprocessing, ClipReward
# from ..utils.tensor_utils import normalize_observation, reshape_observation
# from ..models.dynamics_modeling.transformer_model import StochasticTransformerKVCache
# from ..models.dynamics_modeling.attention_blocks import get_subsequent_mask_with_batch_length


# def gather_steps(env:Env, 
#                  observation:np.ndarray, 
#                  action:int, 
#                  lives:int,
#                  features:torch.Tensor,
#                  state:dict, 
#                  env_steps_per_epoch: int,
#                  actor:Actor, 
#                  encoder:CategoricalEncoder, 
#                  latent_dim:int, 
#                  codes_per_latent:int, 
#                  device:str, 
#                  context_length:int, 
#                  embedding_dim:int, 
#                  storm_transformer:StochasticTransformerKVCache) -> Tuple:
    
#     if 'kv_cache_list' in state:
#         storm_transformer.kv_cache_list = state['kv_cache_list']
    
#     all_observations, all_actions, all_rewards, all_terminations = [], [], [], []

#     action_array = np.zeros(env.action_space.n, dtype=np.float32)
#     action_array[action] = 1.0

#     # random action: a_t
#     # token = (z_t, a_t)
#     # forward = h_t
#     # step: o_t+1, r_t+1, t_t+1 = env.step(a_t)
#     # state = (z_(t+1), h_t)
#     # action: a_t+1
#     # token (z_t+1, a_t+1)
#     # forward: h_t+1
#     # step: 0_t+2, r_t+2, t_t+2 = env.step(a_t+1)
#     # state = (z_t+2, h_t+1)

#     with torch.no_grad():
#         for step in range(env_steps_per_epoch):
#             next_observation, next_reward, next_termination, next_truncated, info = env.step(action) # o_(t+2), r_(t+2), t_(t+2), a_(t+1)

#             all_observations.append(observation) # o_t+1
#             all_actions.append(action_array) # a_t+1
#             all_rewards.append(next_reward) # r_(t+2)

#             # current_lives = info.get("lives", 0)
#             # life_loss = current_lives < lives
#             # lives = current_lives
#             # done = next_termination or life_loss
#             life_loss = info.get("life_loss", False)
#             done = next_termination or life_loss
#             lives = info.get("lives", 0)

#             all_terminations.append(done) # t_(t+2)

#             next_observation = reshape_observation(normalize_observation(observation=next_observation))
#             observation_tensor = torch.from_numpy(next_observation).unsqueeze(0).unsqueeze(0).to(device=device) # o_t+2
#             latent_t = encoder.forward(observations_batch=observation_tensor,
#                                         batch_size=1,
#                                         sequence_length=1,
#                                         latent_dim=latent_dim,
#                                         codes_per_latent=codes_per_latent)
#             latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) # z_t+2

#             env_state_vec = torch.cat([latent_t.view(1, 1, -1), features], dim=-1) # (z_t+2, h_t+1)

#             action_logits = actor(state=env_state_vec)
#             action_idx = torch.argmax(OneHotCategorical(logits=action_logits).sample()).item()
#             action_array = np.zeros(env.action_space.n, dtype=np.float32)
#             action_array[action_idx] = 1.0
#             tensor_action = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device) # a_(t+2)

#             observation = next_observation
#             action = action_idx

#             flattened_sample = latent_t.flatten(start_dim=2)
#             action_tensor_idx = torch.tensor([[action_idx]], device=device)
#             dist_feat = storm_transformer.forward_with_kv_cache(samples=flattened_sample, action=action_tensor_idx)
            
#             if storm_transformer.kv_cache_list[0].shape[1] == context_length:
#                 for idx in range(len(storm_transformer.kv_cache_list)):
#                     storm_transformer.kv_cache_list[idx] = storm_transformer.kv_cache_list[idx][:, 1:, :]

#             # features = features[:, -1:, :]
#             features = dist_feat

#             if next_termination or next_truncated:
#                 observation, info = env.reset()
#                 observation = reshape_observation(normalize_observation(observation=observation)) # o_t

#                 storm_transformer.reset_kv_cache_list(1, dtype=torch.bfloat16)

#                 observation_tensor = torch.from_numpy(observation).unsqueeze(0).unsqueeze(0).to(device=device) # o_t+2
#                 latent_t = encoder.forward(observations_batch=observation_tensor,
#                                             batch_size=1,
#                                             sequence_length=1,
#                                             latent_dim=latent_dim,
#                                             codes_per_latent=codes_per_latent)
#                 latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) # z_t+2

#                 action = env.action_space.sample() # a_t
#                 action_array = np.zeros(env.action_space.n, dtype=np.float32)
#                 action_array[action] = 1.0
#                 tensor_action = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device) # a_(t)

#                 flattened_sample = latent_t.flatten(start_dim=2)
#                 action_tensor_idx_reset = torch.tensor([[action]], device=device)
#                 dist_feat = storm_transformer.forward_with_kv_cache(samples=flattened_sample, action=action_tensor_idx_reset)
#                 # features = features[:, -1:, :]
#                 features = dist_feat

#                 lives = info.get("lives", 0)

                                
#     last_observation = observation
#     observations = np.array(all_observations)
#     actions = np.array(all_actions)
#     rewards = np.array(all_rewards)
#     terminations = np.array(all_terminations)

#     state['kv_cache_list'] = storm_transformer.kv_cache_list
    
#     return observations, actions, rewards, terminations, last_observation, action, lives, features, state

import torch
import numpy as np
from collections import deque
from gymnasium import Env
from typing import Tuple
from scripts.models.agent.actor import Actor
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.sampler import sample
from torch.distributions import OneHotCategorical
from ..utils.tensor_utils import normalize_observation, reshape_observation
from ..models.dynamics_modeling.transformer_model import StochasticTransformerKVCache, DistHead
from ..models.dynamics_modeling.attention_blocks import get_subsequent_mask_with_batch_length


def gather_steps(env: Env,
                 observation: np.ndarray,
                 lives: int,
                 env_steps_per_epoch: int,
                 actor: Actor,
                 encoder: CategoricalEncoder,
                 latent_dim: int,
                 codes_per_latent: int,
                 device: str,
                 storm_transformer: StochasticTransformerKVCache,
                 dist_head: DistHead,
                 context_obs: deque,
                 context_action: deque) -> Tuple:
    """
    Collect environment steps using STORM's context re-encoding approach.
    
    - Maintain context_obs and context_action deques (maxlen=16)
    - Each step: encode full context, run full transformer forward, 
      get prior sample + hidden state, agent selects action
    - No KV cache used (KV cache is only for imagination/dreaming)
    - Context is NOT cleared on episode boundaries
    """
    all_observations, all_actions, all_rewards, all_terminations = [], [], [], []

    with torch.no_grad():
        for step in range(env_steps_per_epoch):
            # === Select action (matches STORM train.py lines 101-115) ===
            if len(context_action) == 0:
                action = env.action_space.sample()
            else:
                # Encode full context observations through encoder
                obs_context = torch.cat(list(context_obs), dim=1)  # (1, T, C, H, W)
                T = obs_context.shape[1]

                latents = encoder.forward(observations_batch=obs_context,
                                          batch_size=1,
                                          sequence_length=T,
                                          latent_dim=latent_dim,
                                          codes_per_latent=codes_per_latent)
                latents_sampled = sample(latents_batch=latents, batch_size=1, sequence_length=T)
                flattened = latents_sampled.flatten(start_dim=2)  # (1, T, latent_dim*codes_per_latent)

                # Stack context actions as index tensor
                act_context = torch.tensor(list(context_action), device=device).unsqueeze(0)  # (1, T)

                # Run full transformer forward (correct positional encoding every time)
                temporal_mask = get_subsequent_mask_with_batch_length(T, device)
                dist_feat = storm_transformer(flattened, act_context, temporal_mask)
                last_dist_feat = dist_feat[:, -1:]  # (1, 1, embedding_dim)

                # Get prior sample from transformer's prediction
                prior_logits = dist_head.forward_prior(last_dist_feat)
                prior_sample = OneHotCategorical(logits=prior_logits).sample()
                prior_flat = prior_sample.flatten(start_dim=2)  # (1, 1, latent_dim*codes_per_latent)

                # Agent selects action from state = (prior_sample, hidden_state)
                env_state = torch.cat([prior_flat, last_dist_feat], dim=-1)
                action_logits = actor(state=env_state)
                action = torch.argmax(OneHotCategorical(logits=action_logits).sample()).item()

            # === Store current observation for dataset ===
            obs_normalized = reshape_observation(normalize_observation(observation))
            all_observations.append(obs_normalized)

            action_array = np.zeros(env.action_space.n, dtype=np.float32)
            action_array[action] = 1.0
            all_actions.append(action_array)

            # === Append to context BEFORE stepping (matches STORM train.py lines 117-118) ===
            obs_tensor = torch.from_numpy(obs_normalized).unsqueeze(0).unsqueeze(0).to(device=device)
            context_obs.append(obs_tensor)
            context_action.append(action)

            # === Step environment ===
            next_observation, next_reward, next_termination, next_truncated, info = env.step(action)

            all_rewards.append(next_reward)

            life_loss = info.get("life_loss", False)
            done = next_termination or life_loss
            all_terminations.append(done)

            # Update observation for next iteration
            observation = next_observation
            lives = info.get("lives", 0)

            # Reset env on episode end (STORM does NOT clear context on reset)
            if next_termination or next_truncated:
                observation, info = env.reset()
                lives = info.get("lives", 0)

    observations = np.array(all_observations)
    actions = np.array(all_actions)
    rewards = np.array(all_rewards)
    terminations = np.array(all_terminations)

    return observations, actions, rewards, terminations, observation, lives