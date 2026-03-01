import torch
import time
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.categorical_vae.sampler import sample
from scripts.models.agent.critic import Critic, critic_loss
from scripts.models.agent.actor import Actor, actor_loss
from scripts.utils.tensor_utils import update_ema_critic
from torch.distributions import OneHotCategorical


def dream(xlstm_dm:XLSTM_DM, 
          tokenizer:Tokenizer, 
          actor:Actor, 
          tokens:torch.Tensor, 
          imagination_horizon:int, 
          latent_dim:int, 
          codes_per_latent:int, 
          batch_size:int, 
          env_actions:int,
          device:str) -> Tuple:
    
    state = None
    context_length = tokens.shape[1]
    for t in range(context_length):
        batch_tokens_t = tokens[:, t:t+1, :]
        latent, reward, termination, feature, state = xlstm_dm.step(tokens_batch=batch_tokens_t, state=state)
    
    imagined_latents = []
    imagined_actions = []
    imagined_rewards = []
    imagined_terminations = []
    features = []

    next_latent = latent.view(batch_size, 1, latent_dim, codes_per_latent)

    for step in range(imagination_horizon):
        next_latent_sample = sample(latents_batch=next_latent, batch_size=batch_size, sequence_length=1)

        imagined_latents.append(next_latent_sample)
        imagined_rewards.append(reward[:, -1, :])
        imagined_terminations.append((termination[:, -1, :] > 0.0).float())

        current_feature = feature[:, -1, :]
        features.append(current_feature)

        flattened_latent = next_latent_sample.view(batch_size, -1)
        env_state = torch.cat([flattened_latent, current_feature], dim=-1)

        action_logits = actor.forward(state=env_state)
        policy = OneHotCategorical(logits=action_logits)
        next_action = policy.sample()
        
        imagined_actions.append(next_action)

        next_token = tokenizer.forward(latents_sampled_batch=next_latent_sample, actions_batch=next_action.unsqueeze(dim=1))

        next_latent, reward, termination, feature, state = xlstm_dm.step(tokens_batch=next_token, state=state)
        next_latent = next_latent.view(batch_size, 1, latent_dim, codes_per_latent)


    imagined_latents = torch.cat(imagined_latents, dim=1)
    imagined_actions = torch.stack(imagined_actions, dim=1)
    imagined_rewards = torch.stack(imagined_rewards, dim=1)
    imagined_terminations = torch.stack(imagined_terminations, dim=1)
    features = torch.stack(features, dim=1)    

    return imagined_latents, imagined_actions, imagined_rewards, imagined_terminations, features


def lambda_returns(reward:torch.Tensor, 
                   termination:torch.Tensor, 
                   gamma:float, 
                   lambda_p:float, 
                   state_value:torch.Tensor, 
                   g_value:torch.Tensor) -> torch.Tensor:
    
    lambda_formula = (1-lambda_p)*state_value + lambda_p*g_value
    return reward + gamma*(1-termination)*lambda_formula


def recursive_lambda_returns(env_state:torch.Tensor, 
                             reward:torch.Tensor, 
                             termination:torch.Tensor, 
                             gamma:float, 
                             lambda_p:float,  
                             device:str, 
                             critic:Critic) -> Tuple:
    
    imagination_horizon = reward.shape[1]
    with torch.no_grad():
        state_values = critic.forward(state=env_state) # (1024, 16, 1)

    batch_lambda_returns = torch.zeros_like(input=state_values, device=device)
    batch_lambda_returns[:, -1, :] = state_values[:, -1, :]

    for timestep in reversed(range(imagination_horizon-1)):
        reward_t = reward[:, timestep, :]
        termination_t = termination[:, timestep, :]
        state_value_t_plus_1 = state_values[:, timestep+1, :]
        g_value_t_plus_1 = batch_lambda_returns[:, timestep+1, :]
        batch_lambda_returns[:, timestep, :] = lambda_returns(reward=reward_t, 
                                                              termination=termination_t, 
                                                              gamma=gamma, 
                                                              lambda_p=lambda_p, 
                                                              state_value=state_value_t_plus_1, 
                                                              g_value=g_value_t_plus_1)
    return batch_lambda_returns.squeeze(-1), state_values.squeeze(-1)


def train_agent(latents_sampled_batch:torch.Tensor, 
                actions_batch:torch.Tensor, 
                context_length:int, 
                imagination_horizon:int, 
                env_actions:int, 
                latent_dim:int, 
                codes_per_latent:int,  
                tokenizer:Tokenizer, 
                xlstm_dm:XLSTM_DM, 
                actor:Actor, 
                critic:Critic, 
                ema_critic:Critic,
                device:str, 
                gamma:float, 
                lambda_p: float, 
                ema_sigma:float, 
                nabla:float, 
                optimizer:torch.optim.Adam, 
                scaler:torch.amp.GradScaler) -> Tuple:

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            latents_sampled_batch = latents_sampled_batch.view(-1, context_length, latent_dim*codes_per_latent)
            actions_batch = actions_batch.view(-1, context_length, env_actions)
            tokens_batch = tokenizer.forward(latents_sampled_batch=latents_sampled_batch, actions_batch=actions_batch)

            imagined_latent, imagined_action, imagined_reward, imagined_termination, feature = dream(xlstm_dm=xlstm_dm, 
                                                                                                    tokenizer=tokenizer, 
                                                                                                    actor=actor, 
                                                                                                    tokens=tokens_batch, 
                                                                                                    imagination_horizon=imagination_horizon, 
                                                                                                    latent_dim=latent_dim, 
                                                                                                    codes_per_latent=codes_per_latent, 
                                                                                                    batch_size=tokens_batch.shape[0], 
                                                                                                    env_actions=env_actions, 
                                                                                                    device=device)

            env_state = torch.concat([imagined_latent, feature], dim=-1)
            regular_lambda_returns, _ = recursive_lambda_returns(env_state=env_state, 
                                                                            reward=imagined_reward, 
                                                                            termination=imagined_termination, 
                                                                            gamma=gamma, 
                                                                            lambda_p=lambda_p, 
                                                                            device=device, 
                                                                            critic=critic)
            
            ema_lambda_returns, _ = recursive_lambda_returns(env_state=env_state, 
                                                            reward=imagined_reward, 
                                                            termination=imagined_termination, 
                                                            gamma=gamma, 
                                                            lambda_p=lambda_p, 
                                                            device=device, 
                                                            critic=ema_critic)

    state_values = critic.forward(state=env_state).squeeze(-1)

    action_logits = actor.forward(state=env_state.detach())

    policy = OneHotCategorical(logits=action_logits)

    log_policy = policy.log_prob(imagined_action.detach())

    entropy = policy.entropy()
    
    mean_actor_loss = actor_loss(batch_lambda_returns=regular_lambda_returns, 
                                    state_values=state_values, 
                                    log_policy=log_policy, 
                                    nabla=nabla, 
                                    entropy=entropy)
    
    mean_critic_loss = critic_loss(batch_lambda_returns=regular_lambda_returns, 
                                    state_values=state_values, 
                                    ema_lambda_returns=ema_lambda_returns)
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(mean_actor_loss).backward()
    scaler.scale(mean_critic_loss).backward()
    scaler.unscale_(optimizer)
    
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    update_ema_critic(ema_sigma=ema_sigma, critic=critic, ema_critic=ema_critic)

    mean_imagined_reward = imagined_reward.mean().item()
    return mean_actor_loss.item(), mean_critic_loss.item(), mean_imagined_reward