import os
import torch
import numpy as np
from collections import deque
from typing import Tuple
from ..world_model.transformer.transformer import TransformerDecoder
from ..world_model.categorical_autoencoder.encoder import CategoricalEncoder
from ..world_model.categorical_autoencoder.decoder import CategoricalDecoder
from ..world_model.categorical_autoencoder.sampler import sample
from ..world_model.transformer.latent_action_embedder import LatentActionEmbedder
from ..world_model.transformer.dynamics_step import SymLogTwoHotLoss
from scripts.models.agent.critic import Critic, critic_loss
from scripts.models.agent.actor import Actor, actor_loss
from scripts.utils.tensor_utils import update_ema_critic
from torch.distributions import OneHotCategorical
from scripts.utils.tensor_utils import percentile, EMAScalar
from scripts.utils.debug_utils import save_rollout_video
import torch.nn.functional as F


def take_action(context_obs:deque, 
                context_act:deque, 
                categorical_encoder:CategoricalEncoder, 
                latent_action_embedder:LatentActionEmbedder, 
                transformer:TransformerDecoder, 
                actor:Actor, 
                latent_dim:int, 
                codes_per_latent:int, 
                env_actions:int, 
                device:str, 
                tensor_dtype) -> int:
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.stack(list(context_obs))).unsqueeze(0).to(device).to(tensor_dtype) / 255.0
            act_tensor = torch.from_numpy(np.stack(list(context_act))).unsqueeze(0).to(device).float()
            ctx_len = obs_tensor.shape[1]

            posterior_raw_logits = categorical_encoder.forward(observations_batch=obs_tensor, 
                                                            batch_size=1, 
                                                            sequence_length=ctx_len, 
                                                            latent_dim=latent_dim, 
                                                            codes_per_latent=codes_per_latent)
            posterior_sample, posterior_logits = sample(posterior_raw_logits=posterior_raw_logits)

            raw_actions = act_tensor.long()
            actions_batch = F.one_hot(raw_actions, num_classes=env_actions).to(torch.uint8)
            latent_action_embeddings = latent_action_embedder.forward(posterior_sample_batch=posterior_sample, actions_batch=actions_batch, start_pos=0)

            mask = torch.tril(torch.ones((ctx_len, ctx_len), device=device)).unsqueeze(0).unsqueeze(0)
            prior_raw_logits, reward_logits, termination_logits, x, _ = transformer(x=latent_action_embeddings, mask=mask) # z_1, h_0

            prior_raw_logits = prior_raw_logits[:, -1:, :]
            x = x[:, -1:, :]

            prior_sample_batch, prior_logits = sample(posterior_raw_logits=prior_raw_logits.view(1, 1, latent_dim, codes_per_latent)) # z_1

            prior_sample_batch_flattened = prior_sample_batch.view(1, -1)
            env_state = torch.cat([prior_sample_batch_flattened, x.squeeze(dim=1)], dim=-1) # z_1, h_0
            action_logits = actor.forward(state=env_state)
            policy = OneHotCategorical(logits=action_logits)
            action_idx = torch.argmax(policy.sample()).item() # a_1
            return action_idx


def dream(transformer:TransformerDecoder, 
          categorical_encoder:CategoricalEncoder, 
          categorical_decoder:CategoricalDecoder, 
          latent_action_embedder:LatentActionEmbedder, 
          actor:Actor, 
          observations_batch:torch.Tensor, 
          actions_batch:torch.Tensor, 
          batch_size:int, 
          context_length:int, 
          latent_dim:int, 
          codes_per_latent:int, 
          imagination_horizon:int, 
          save_video:bool, 
          video_path:str, 
          total_env_steps:int) -> Tuple:
    
        posterior_raw_logits = categorical_encoder.forward(observations_batch=observations_batch, 
                                                        batch_size=batch_size, 
                                                        sequence_length=context_length, 
                                                        latent_dim=latent_dim, 
                                                        codes_per_latent=codes_per_latent)    
        posterior_sample, posterior_logits = sample(posterior_raw_logits=posterior_raw_logits)

        latent_action_embeddings = latent_action_embedder.forward(posterior_sample_batch=posterior_sample, actions_batch=actions_batch, start_pos=0)

        mask = torch.tril(torch.ones((context_length, context_length), device='cuda'))
        mask = mask.unsqueeze(0).unsqueeze(0)
        transformer.reset_kv_cache()
        prior_raw_logits, reward_logits, termination_logits, x = transformer.forward_kv_cache(x=latent_action_embeddings, mask=mask) 
        x = x[:, -1:, :]
        prior_raw_logits = prior_raw_logits[:, -1:, :] 

        imagined_latents = []
        imagined_actions = []
        imagined_rewards = []
        imagined_terminations = []
        features = []
        for step in range(imagination_horizon):
            features.append(x.squeeze(1))

            prior_sample_batch, prior_logits = sample(posterior_raw_logits=prior_raw_logits.view(batch_size, 1, latent_dim, codes_per_latent)) # z_1

            prior_sample_batch_flattened = prior_sample_batch.view(batch_size, -1)
            env_state = torch.cat([prior_sample_batch_flattened, x.squeeze(dim=1)], dim=-1).detach()
            action_logits = actor.forward(state=env_state)
            policy = OneHotCategorical(logits=action_logits)
            next_action = policy.sample()
    
            current_position = context_length + step
            latent_action_embeddings = latent_action_embedder.forward(posterior_sample_batch=prior_sample_batch, actions_batch=next_action.unsqueeze(dim=1), start_pos=current_position)

            prior_raw_logits, reward_logits, termination_logits, x = transformer.forward_kv_cache(x=latent_action_embeddings, mask=None) # z_2, r_1, t_1

            imagined_latents.append(prior_sample_batch)
            imagined_actions.append(next_action)
            imagined_rewards.append(reward_logits.squeeze(1))
            imagined_terminations.append((termination_logits.squeeze(1) > 0.0).float())

        if save_video:
            with torch.no_grad():
                num_videos = 9
                rollout_idx = torch.randperm(batch_size, device=observations_batch.device)[:num_videos]
                latents_stacked = torch.cat(imagined_latents, dim=1)
                selected = latents_stacked[rollout_idx]
                frames = categorical_decoder.forward(posterior_sample=selected, 
                                                     batch_size=num_videos, 
                                                     sequence_length=imagination_horizon, 
                                                     latent_dim=latent_dim, 
                                                     codes_per_latent=codes_per_latent)
                save_rollout_video(frames, video_path, total_env_steps)

        final_prior_sample, _ = sample(posterior_raw_logits=prior_raw_logits.view(batch_size, 1, latent_dim, codes_per_latent))
        imagined_latents.append(final_prior_sample)
        features.append(x.squeeze(1))

        imagined_latents = torch.cat(imagined_latents, dim=1)
        imagined_actions = torch.stack(imagined_actions, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_terminations = torch.stack(imagined_terminations, dim=1)
        features = torch.stack(features, dim=1)    

        transformer.reset_kv_cache()

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
                             critic:Critic, 
                             symlog_twohot_loss_func:SymLogTwoHotLoss) -> Tuple:
    
    imagination_horizon = reward.shape[1]
    with torch.no_grad():
        state_values = critic.forward(state=env_state)
        state_values = symlog_twohot_loss_func.decode(state_values)
        reward = symlog_twohot_loss_func.decode(reward)

    batch_lambda_returns = torch.zeros_like(input=state_values, device=device)
    batch_lambda_returns[:, -1] = state_values[:, -1]

    for timestep in reversed(range(imagination_horizon)):
        reward_t = reward[:, timestep]
        termination_t = termination[:, timestep].view(-1)
        state_value_t = state_values[:, timestep]
        g_value_t_plus_1 = batch_lambda_returns[:, timestep+1]
        batch_lambda_returns[:, timestep] = lambda_returns(reward=reward_t, 
                                                           termination=termination_t, 
                                                           gamma=gamma, 
                                                           lambda_p=lambda_p, 
                                                           state_value=state_value_t, 
                                                           g_value=g_value_t_plus_1)
        
    return batch_lambda_returns, state_values


def train_agent(observations_batch:torch.Tensor, 
                actions_batch:torch.Tensor, 
                context_length:int, 
                imagination_horizon:int, 
                latent_dim:int, 
                codes_per_latent:int, 
                agent_batch_size:int,  
                categorical_encoder:CategoricalEncoder, 
                categorical_decoder:CategoricalDecoder, 
                transformer:TransformerDecoder, 
                latent_action_embedder:LatentActionEmbedder, 
                actor:Actor, 
                critic:Critic, 
                ema_critic:Critic,
                device:str, 
                gamma:float, 
                lambda_p: float, 
                ema_sigma:float, 
                nabla:float, 
                optimizer:torch.optim.Adam, 
                scaler:torch.amp.GradScaler, 
                lowerbound_ema:EMAScalar,
                upperbound_ema:EMAScalar, 
                save_video:bool, 
                run_dir:str, 
                env_step:int) -> Tuple:
    
    symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20).to(device='cuda')

    actor.eval()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            imagined_latent, imagined_action, imagined_reward, imagined_termination, feature = dream(transformer=transformer, 
                                                                                                     categorical_encoder=categorical_encoder, 
                                                                                                     categorical_decoder=categorical_decoder, 
                                                                                                     latent_action_embedder=latent_action_embedder, 
                                                                                                     actor=actor, 
                                                                                                     observations_batch=observations_batch, 
                                                                                                     actions_batch=actions_batch, 
                                                                                                     batch_size=agent_batch_size, 
                                                                                                     context_length=context_length, 
                                                                                                     latent_dim=latent_dim, 
                                                                                                     codes_per_latent=codes_per_latent, 
                                                                                                     imagination_horizon=imagination_horizon, 
                                                                                                     save_video=save_video, 
                                                                                                     video_path=os.path.join(run_dir, "videos"), 
                                                                                                     total_env_steps=env_step)

            env_state = torch.concat([torch.flatten(imagined_latent, start_dim=2), feature], dim=-1)
            regular_lambda_returns, _ = recursive_lambda_returns(env_state=env_state, 
                                                                            reward=imagined_reward, 
                                                                            termination=imagined_termination, 
                                                                            gamma=gamma, 
                                                                            lambda_p=lambda_p, 
                                                                            device=device, 
                                                                            critic=critic, 
                                                                            symlog_twohot_loss_func=symlog_twohot_loss_func)
            
            ema_lambda_returns, _ = recursive_lambda_returns(env_state=env_state, 
                                                            reward=imagined_reward, 
                                                            termination=imagined_termination, 
                                                            gamma=gamma, 
                                                            lambda_p=lambda_p, 
                                                            device=device, 
                                                            critic=ema_critic, 
                                                            symlog_twohot_loss_func=symlog_twohot_loss_func)

    critic.train()
    actor.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        state_logits = critic.forward(state=env_state)
        state_values = symlog_twohot_loss_func.decode(state_logits)

        action_logits = actor.forward(state=env_state.detach())

        policy = OneHotCategorical(logits=action_logits[:, :-1])

        log_policy = policy.log_prob(imagined_action.detach())

        entropy = policy.entropy()

        lower_bound = lowerbound_ema(percentile(regular_lambda_returns[:, :-1], 0.05))
        upper_bound = upperbound_ema(percentile(regular_lambda_returns[:, :-1], 0.95))
        S = upper_bound - lower_bound
        norm_ratio = torch.max(torch.ones(1, device=device), S)
        
        mean_actor_loss = actor_loss(batch_lambda_returns=regular_lambda_returns[:, :-1], 
                                        state_values=state_values[:, :-1], 
                                        log_policy=log_policy, 
                                        nabla=nabla, 
                                        entropy=entropy, 
                                        norm_ratio=norm_ratio)
        
        mean_critic_loss = critic_loss(batch_lambda_returns=regular_lambda_returns[:, :-1], 
                                        state_values=state_logits[:, :-1], 
                                        ema_lambda_returns=ema_lambda_returns[:, :-1], 
                                        symlog_twohot_loss=symlog_twohot_loss_func)
        
        total_loss = mean_actor_loss + mean_critic_loss
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    
    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 100.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    update_ema_critic(ema_sigma=ema_sigma, critic=critic, ema_critic=ema_critic)

    return mean_actor_loss.item(), mean_critic_loss.item(), entropy[:, :-1].mean().item(), S.item(), norm_ratio.item()