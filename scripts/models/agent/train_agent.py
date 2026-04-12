import torch
from typing import Tuple
from scripts.models.dynamics_modeling.encoder import CategoricalEncoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.dynamics_model_step import SymLogTwoHotLoss
from scripts.models.dynamics_modeling.sampler import sample
from scripts.models.agent.critic import Critic, critic_loss
from scripts.models.agent.actor import Actor, actor_loss
from scripts.utils.tensor_utils import update_ema_critic
from torch.distributions import OneHotCategorical
from scripts.utils.tensor_utils import percentile, EMAScalar

def dream(storm_transformer:StochasticTransformerKVCache, 
          dist_head:DistHead, 
          reward_decoder:RewardDecoder, 
          termination_decoder:TerminationDecoder,
          actor:Actor, 
          imagination_horizon:int, 
          latents_sampled_batch:torch.Tensor, 
          actions_indices:torch.Tensor, 
          batch_size:int, 
          device:str) -> Tuple:
    
    storm_transformer.reset_kv_cache_list(batch_size, dtype=torch.bfloat16)
    flattened_latents = latents_sampled_batch.flatten(start_dim=2)

    with torch.no_grad():
            for i in range(flattened_latents.shape[1]):
                dist_feat = storm_transformer.forward_with_kv_cache(
                    samples=flattened_latents[:, i:i+1, :], 
                    action=actions_indices[:, i:i+1]
            )

    h_t = dist_feat 
    prior_logits = dist_head.forward_prior(h_t)
    latent_pred = prior_logits 
        
    imagined_latents = []
    imagined_actions = []
    imagined_rewards = []
    imagined_terminations = []
    features = []

    for step in range(imagination_horizon):
        next_latent_sample = sample(latents_batch=latent_pred, batch_size=batch_size, sequence_length=1)
        imagined_latents.append(next_latent_sample)

        current_feature = h_t.squeeze(1)
        features.append(current_feature)

        next_latent_sample_flattened = next_latent_sample.flatten(start_dim=2)
        env_state = torch.cat([next_latent_sample_flattened.squeeze(1), current_feature], dim=-1).detach()

        action_logits = actor.forward(state=env_state)
        policy = OneHotCategorical(logits=action_logits)
        next_action_onehot = policy.sample()
        next_action_idx = torch.argmax(next_action_onehot, dim=-1).unsqueeze(1)
        
        imagined_actions.append(next_action_onehot)

        with torch.no_grad():
            dist_feat = storm_transformer.forward_with_kv_cache(
                samples=next_latent_sample_flattened, 
                action=next_action_idx
            )
            prior_logits = dist_head.forward_prior(dist_feat)
            reward_pred = reward_decoder(dist_feat)
            term_pred = termination_decoder(dist_feat)
            
        latent_pred = prior_logits
        h_t = dist_feat

        imagined_rewards.append(reward_pred.squeeze(1))
        imagined_terminations.append((term_pred.squeeze(1) > 0.0).float())

    final_sample = sample(latents_batch=latent_pred, batch_size=batch_size, sequence_length=1)
    imagined_latents.append(final_sample)
    features.append(h_t.squeeze(1))


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
                env_actions:int, 
                latent_dim:int, 
                codes_per_latent:int, 
                agent_batch_size:int,  
                categorical_encoder:CategoricalEncoder, 
                storm_transformer:StochasticTransformerKVCache, 
                dist_head:DistHead, 
                reward_decoder:RewardDecoder, 
                termination_decoder:TerminationDecoder,
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
                upperbound_ema:EMAScalar) -> Tuple:
    
    symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20).to(device='cuda')

    categorical_encoder.eval()
    storm_transformer.eval()
    dist_head.eval()
    reward_decoder.eval()
    termination_decoder.eval()
    actor.eval()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            latents_batch = categorical_encoder.forward(observations_batch=observations_batch, 
                                                        batch_size=agent_batch_size, 
                                                        sequence_length=context_length, 
                                                        latent_dim=latent_dim, 
                                                        codes_per_latent=codes_per_latent)    

            latents_sampled_batch = sample(latents_batch=latents_batch, batch_size=agent_batch_size, sequence_length=context_length)

            actions_batch = actions_batch.view(-1, context_length, env_actions)
            if actions_batch.dim() == 3:
                actions_indices = torch.argmax(actions_batch, dim=-1)
            else:
                actions_indices = actions_batch

            imagined_latent, imagined_action, imagined_reward, imagined_termination, feature = dream(storm_transformer=storm_transformer,
                                                                                                     dist_head=dist_head, 
                                                                                                     reward_decoder=reward_decoder, 
                                                                                                     termination_decoder=termination_decoder,
                                                                                                    actor=actor, 
                                                                                                    latents_sampled_batch=latents_sampled_batch, 
                                                                                                    actions_indices=actions_indices, 
                                                                                                    imagination_horizon=imagination_horizon, 
                                                                                                    batch_size=latents_sampled_batch.shape[0],
                                                                                                    device=device)

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