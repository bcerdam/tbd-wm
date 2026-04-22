import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from ..models.world_model.transformer.latent_action_embedder import LatentActionEmbedder
from ..models.world_model.categorical_autoencoder.encoder import CategoricalEncoder
from ..models.world_model.categorical_autoencoder.decoder import CategoricalDecoder
from ..models.world_model.categorical_autoencoder.sampler import sample
from ..models.world_model.transformer.transformer import TransformerDecoder


def tensorboard_update(writer,
                       total_env_steps, 
                       world_model_loss, 
                       reconstruction_loss, 
                       rewards_loss, 
                       terminations_loss, 
                       dynamics_loss, 
                       dynamics_real_kl_div, 
                       representation_loss, 
                       representation_real_kl_div) -> None:
    writer.add_scalar('loss/total', world_model_loss.item(), total_env_steps)
    writer.add_scalar('loss/reconstruction', reconstruction_loss.item(), total_env_steps)
    writer.add_scalar('loss/reward', rewards_loss.item(), total_env_steps)
    writer.add_scalar('loss/termination', terminations_loss.item(), total_env_steps)
    writer.add_scalar('loss/dynamics', dynamics_loss.item(), total_env_steps)
    writer.add_scalar('loss/representation', representation_loss.item(), total_env_steps)
    writer.add_scalar('kl/dynamics', dynamics_real_kl_div.item(), total_env_steps)
    writer.add_scalar('kl/representation', representation_real_kl_div.item(), total_env_steps)


def save_checkpoint(encoder, decoder, storm_transformer, dist_head, 
                    reward_decoder, termination_decoder, 
                    actor, critic, ema_critic,
                    wm_optimizer, agent_optimizer, 
                    scaler, step, path="output/checkpoints"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'step': step,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'storm_transformer': storm_transformer.state_dict(),
        'dist_head': dist_head.state_dict(),
        'reward_decoder': reward_decoder.state_dict(),
        'termination_decoder': termination_decoder.state_dict(),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'ema_critic': ema_critic.state_dict(),
        'wm_optimizer': wm_optimizer.state_dict(),
        'agent_optimizer': agent_optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }, os.path.join(path, f"checkpoint_step_{step}.pth"))


def save_rollout_video(frames, output_dir, env_step, fps=15):
    """
    frames: (9, horizon, C, H, W) in [0, 1]
    Arranges as 3x3 grid and saves as mp4.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frames = frames.float().cpu().numpy()
    frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    # (9, H, C, height, width) -> (9, H, height, width, C)
    frames = np.transpose(frames, (0, 1, 3, 4, 2))
    
    num_videos, horizon, h, w, c = frames.shape
    grid_h, grid_w = h * 3, w * 3
    
    path = os.path.join(output_dir, f"rollouts_step_{env_step}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (grid_w, grid_h))
    
    for t in range(horizon):
        grid = np.zeros((grid_h, grid_w, c), dtype=np.uint8)
        for i in range(9):
            row, col = i // 3, i % 3
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = frames[i, t]
        if c == 3:
            grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        writer.write(grid)
    
    writer.release()


def dream(transformer:TransformerDecoder, 
          categorical_encoder:CategoricalEncoder, 
          categorical_decoder:CategoricalDecoder, 
          latent_action_embedder:LatentActionEmbedder, 
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
    
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
            features.append(x)

            prior_sample_batch, posterior_logits = sample(posterior_raw_logits=prior_raw_logits.view(batch_size, 1, latent_dim, codes_per_latent)) # z_1

            num_actions = actions_batch.shape[-1]
            batch_sz = prior_sample_batch.shape[0]
            random_action_idx = torch.randint(0, num_actions, (batch_sz,), device=actions_batch.device)
            sampled_one_hot = F.one_hot(random_action_idx, num_classes=num_actions).unsqueeze(1).float()

            current_position = context_length + step
            latent_action_embeddings = latent_action_embedder.forward(posterior_sample_batch=prior_sample_batch, actions_batch=sampled_one_hot, start_pos=current_position)

            prior_raw_logits, reward_logits, termination_logits, x = transformer.forward_kv_cache(x=latent_action_embeddings, mask=None) # z_2, r_1, t_1

            imagined_latents.append(prior_sample_batch)
            imagined_actions.append(sampled_one_hot)
            imagined_rewards.append(reward_logits)
            imagined_terminations.append((termination_logits > 0.0).float())

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

        imagined_latents = torch.cat(imagined_latents, dim=1)
        imagined_actions = torch.stack(imagined_actions, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_terminations = torch.stack(imagined_terminations, dim=1)
        features = torch.stack(features, dim=1)    

        transformer.reset_kv_cache()

        return imagined_latents, imagined_actions, imagined_rewards, imagined_terminations, features