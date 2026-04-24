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
                       representation_real_kl_div, 
                       actor_loss, 
                       critic_loss, 
                       total_agent_loss, 
                       entropy, 
                       S, 
                       norm_ratio, 
                       episode_mean_rewards) -> None:
    writer.add_scalar('loss/total', world_model_loss.item(), total_env_steps)
    writer.add_scalar('loss/reconstruction', reconstruction_loss.item(), total_env_steps)
    writer.add_scalar('loss/reward', rewards_loss.item(), total_env_steps)
    writer.add_scalar('loss/termination', terminations_loss.item(), total_env_steps)
    writer.add_scalar('loss/dynamics', dynamics_loss.item(), total_env_steps)
    writer.add_scalar('loss/representation', representation_loss.item(), total_env_steps)
    writer.add_scalar('kl/dynamics', dynamics_real_kl_div.item(), total_env_steps)
    writer.add_scalar('kl/representation', representation_real_kl_div.item(), total_env_steps)
    writer.add_scalar('agent/actor', actor_loss, total_env_steps)
    writer.add_scalar('agent/critic', critic_loss, total_env_steps)
    writer.add_scalar('agent/total_loss', total_agent_loss, total_env_steps)
    writer.add_scalar('agent/entropy', entropy, total_env_steps)
    writer.add_scalar('agent/S', S, total_env_steps)
    writer.add_scalar('agent/norm_ratio', norm_ratio, total_env_steps)
    if episode_mean_rewards is not None:
        writer.add_scalar('agent/episode_mean_rewards', episode_mean_rewards, total_env_steps)


def save_checkpoint(categorical_encoder, 
                    categorical_decoder, 
                    latent_action_embedder, 
                    transformer, 
                    actor, 
                    critic, 
                    ema_critic,
                    wm_optimizer, 
                    agent_optimizer, 
                    wm_scaler, 
                    agent_scaler, 
                    step, 
                    path="output/run/checkpoints"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'step': step,
        'categorical_encoder': categorical_encoder.state_dict(),
        'categorical_decoder': categorical_decoder.state_dict(),
        'latent_action_embedder': latent_action_embedder.state_dict(),
        'transformer': transformer.state_dict(),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'ema_critic': ema_critic.state_dict(),
        'wm_optimizer': wm_optimizer.state_dict(),
        'agent_optimizer': agent_optimizer.state_dict(),
        'wm_scaler': wm_scaler.state_dict(), 
        'agent_scaler': agent_scaler.state_dict()
    }, os.path.join(path, f"checkpoint_step_{step}.pth"))


def save_rollout_video(frames, output_dir, env_step, writer, fps=15):
    # """
    # frames: (9, horizon, C, H, W) in [0, 1]
    # Arranges as 3x3 grid and saves as mp4.
    # """
    # os.makedirs(output_dir, exist_ok=True)
    
    # frames = frames.float().cpu().numpy()
    # frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    # # (9, H, C, height, width) -> (9, H, height, width, C)
    # frames = np.transpose(frames, (0, 1, 3, 4, 2))
    
    # num_videos, horizon, h, w, c = frames.shape
    # grid_h, grid_w = h * 3, w * 3
    
    # path = os.path.join(output_dir, f"rollouts_step_{env_step}.mp4")
    # writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (grid_w, grid_h))
    
    # for t in range(horizon):
    #     grid = np.zeros((grid_h, grid_w, c), dtype=np.uint8)
    #     for i in range(9):
    #         row, col = i // 3, i % 3
    #         grid[row*h:(row+1)*h, col*w:(col+1)*w] = frames[i, t]
    #     if c == 3:
    #         grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    #     writer.write(grid)
    
    # writer.release()
    clean_frames = torch.clamp(frames.detach().cpu().float(), 0.0, 1.0)
    
    # 3. Add to TensorBoard (It will handle the batch of 9 videos automatically)
    # Note: TensorBoard will display these in the "IMAGES" tab!
    writer.add_video('Imagine/predict_video', clean_frames, env_step, fps=fps)
