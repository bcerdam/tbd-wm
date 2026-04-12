import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

import cv2
import torch
import sys
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Dict, Union
# from scripts.models.dynamics_modeling.sampler import sample
# from scripts.models.dynamics_modeling.encoder import CategoricalEncoder
# from scripts.models.dynamics_modeling.decoder import CategoricalDecoder
from gymnasium.wrappers import AtariPreprocessing
from scripts.utils.tensor_utils import normalize_observation, reshape_observation
from scripts.data_related.atari_dataset import AtariDataset



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


def save_loss_history(new_losses: List[Dict[str, float]], output_dir: str) -> None:
    keys = new_losses[0].keys()
    epoch_means = {}
    for k in keys:
        valid_vals = [d[k] for d in new_losses if d[k] is not None]
        epoch_means[k] = np.mean(valid_vals) if valid_vals else np.nan

    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, 'loss_history.npy')

    loss_history = np.load(history_path, allow_pickle=True).item() if os.path.exists(history_path) else {}

    for k, v in epoch_means.items():
        loss_history.setdefault(k, []).append(v)

    np.save(history_path, loss_history)


def plot_current_loss(training_steps_per_epoch: int, epochs: int, output_dir: str) -> None:
    history_path = os.path.join(output_dir, 'loss_history.npy')
    loss_history = np.load(history_path, allow_pickle=True).item()

    current_epoch = len(loss_history['reconstruction'])
    if current_epoch == 0: #
        return #
        
    x_values = np.arange(1, current_epoch + 1) * training_steps_per_epoch

    metrics_to_plot = [
        ('reconstruction', 'Reconstruction Loss'),
        ('reward', 'Reward Loss'),
        ('termination', 'Termination Loss'),
        ('dynamics', 'Dynamics Loss'),
        ('dynamics_kl_div', 'Dynamics KL Div'), 
        ('representation', 'Representation Loss'), 
        ('representation_kl_div', 'Representation KL Div'),
        ('actor', 'Actor Loss'),
        ('critic', 'Critic Loss'),
        ('entropy', 'Entropy'),
        ('S', 'S Value'), 
        ('norm_ratio', 'Norm Ratio'),
        ('mean_episode_reward', 'Mean Episode Reward')
    ]

    fig, axes = plt.subplots(nrows=len(metrics_to_plot), ncols=1, figsize=(6, 2.5 * len(metrics_to_plot)), dpi=200, sharex=True)

    for ax, (key, title) in zip(axes, metrics_to_plot):
        if key in loss_history:
            y_vals = np.array(loss_history[key], dtype=float)
            mask = ~np.isnan(y_vals)
            if mask.any():
                if key == 'mean_episode_reward':
                    ax.plot(x_values[mask], y_vals[mask], color='black', linewidth=0.5, marker='o', markersize=2, label=title)
                else:
                    ax.plot(x_values[mask], y_vals[mask], color='black', linewidth=0.5, alpha=0.9, label=title)
                ax.legend(fontsize=5, loc='upper right', framealpha=0.8) #
            
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.set_ylabel("Value", fontsize=6)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=5)

    axes[-1].set_xlabel("Total Training Steps", fontsize=6)
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:g}K')) #
    axes[-1].set_xlim(left=x_values[0], right=x_values[-1] if len(x_values) > 1 else None) #

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.jpeg'), format='jpeg', dpi=200, bbox_inches='tight')
    plt.close()


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


def visualize_reconstruction(env_name: str,
                             weights_path: str, 
                             device: str, 
                             sequence_length: int, 
                             latent_dim: int, 
                             codes_per_latent: int, 
                             epoch: int,
                             video_path: str = "output/videos") -> None:

    os.makedirs(video_path, exist_ok=True)

    encoder = CategoricalEncoder(latent_dim=latent_dim, codes_per_latent=codes_per_latent).to(device)
    decoder = CategoricalDecoder(latent_dim=latent_dim, codes_per_latent=codes_per_latent).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    
    def clean_state_dict(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    encoder.load_state_dict(clean_state_dict(checkpoint['encoder']))
    decoder.load_state_dict(clean_state_dict(checkpoint['decoder']))

    encoder.eval()
    decoder.eval()

    gym.register_envs(ale_py)
    env = gym.make(id=env_name, frameskip=1)
    env = FireOnLifeLossWrapper(env)
    env = AtariPreprocessing(env=env, noop_max=30, frame_skip=4, screen_size=64, terminal_on_life_loss=False, grayscale_obs=False)

    obs_seq = []
    obs, _ = env.reset()
    for _ in range(sequence_length):
        processed_obs = reshape_observation(normalize_observation(observation=obs))
        obs_seq.append(processed_obs)
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    model_input = torch.from_numpy(np.array(obs_seq)).unsqueeze(0).to(device)

    with torch.no_grad():
        latents = encoder.forward(observations_batch=model_input, batch_size=1, sequence_length=sequence_length, latent_dim=latent_dim, codes_per_latent=codes_per_latent)
        sampled_latents = sample(latents_batch=latents, batch_size=1, sequence_length=sequence_length)
        reconstructions = decoder.forward(latents_batch=sampled_latents, batch_size=1, sequence_length=sequence_length, latent_dim=latent_dim, codes_per_latent=codes_per_latent)

    model_input = model_input.cpu()
    reconstructions = reconstructions.cpu()

    orig_np = ((model_input[0].permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    recon_np = ((reconstructions[0].permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

    save_file = os.path.join(video_path, f"epoch_{epoch}_reconstruction.mp4")
    height, width = orig_np.shape[1], orig_np.shape[2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, 15.0, (width * 2, height))

    for t in range(sequence_length):
        frame_orig = cv2.cvtColor(orig_np[t], cv2.COLOR_RGB2BGR)
        frame_recon = cv2.cvtColor(recon_np[t], cv2.COLOR_RGB2BGR)
        combined_frame = np.concatenate((frame_orig, frame_recon), axis=1)
        out.write(combined_frame)

    out.release()


def save_real_video(imagined_frames: List[np.ndarray], 
                     imagined_rewards: List[Union[torch.Tensor, np.ndarray]], 
                     imagined_terminations: List[Union[torch.Tensor, np.ndarray]], 
                     video_path: str, 
                     fps: int) -> None:
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    _, _, orig_height, orig_width = imagined_frames[0].shape
    scale = 4
    height, width = orig_height * scale, orig_width * scale

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame, reward, term in zip(imagined_frames, imagined_rewards, imagined_terminations):
        frame = frame[0]
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame + 1) * 127.5
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.shape[-1] == 1 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        r_val = reward[0].item() if hasattr(reward[0], 'item') else float(reward[0])
        t_val = term[0].item() if hasattr(term[0], 'item') else float(term[0])

        cv2.putText(frame, f"R: {r_val:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"T: {t_val:.3f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()


def save_dream_video(real_frames: List[np.ndarray],
                     imagined_frames: List[np.ndarray], 
                     real_rewards: List[Union[torch.Tensor, np.ndarray]],
                     imagined_rewards: List[Union[torch.Tensor, np.ndarray]],
                     real_terminations: List[Union[torch.Tensor, np.ndarray]],
                     imagined_terminations: List[Union[torch.Tensor, np.ndarray]],
                     video_path: str, 
                     fps: int) -> None:
    
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    _, _, orig_height, orig_width = imagined_frames[0].shape
    scale = 4
    height, width = orig_height * scale, orig_width * scale
    double_width = width * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (double_width, height))

    for r_f, i_f, r_rew, i_rew, r_term, i_term in zip(real_frames, imagined_frames, real_rewards, imagined_rewards, real_terminations, imagined_terminations):
        
        r_img = np.transpose(r_f[0], (1, 2, 0))
        r_img = np.clip((r_img + 1) * 127.5, 0, 255).astype(np.uint8)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_GRAY2BGR) if r_img.shape[-1] == 1 else cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
        r_img = cv2.resize(r_img, (width, height), interpolation=cv2.INTER_NEAREST)

        i_img = np.transpose(i_f[0], (1, 2, 0))
        i_img = np.clip((i_img + 1) * 127.5, 0, 255).astype(np.uint8)
        i_img = cv2.cvtColor(i_img, cv2.COLOR_GRAY2BGR) if i_img.shape[-1] == 1 else cv2.cvtColor(i_img, cv2.COLOR_RGB2BGR)
        i_img = cv2.resize(i_img, (width, height), interpolation=cv2.INTER_NEAREST)

        r_r_val = r_rew[0].item() if hasattr(r_rew[0], 'item') else float(r_rew[0])
        r_t_val = r_term[0].item() if hasattr(r_term[0], 'item') else float(r_term[0])
        
        i_r_val = i_rew[0].item() if hasattr(i_rew[0], 'item') else float(i_rew[0])
        i_t_val = i_term[0].item() if hasattr(i_term[0], 'item') else float(i_term[0])

        cv2.putText(r_img, f"Real R: {r_r_val:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(r_img, f"Real T: {r_t_val:.3f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(i_img, f"Imag R: {i_r_val:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(i_img, f"Imag T: {i_t_val:.3f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        combined_frame = np.concatenate((r_img, i_img), axis=1)
        out.write(combined_frame)

    out.release()


def generate_dataset_video(dataset: AtariDataset, output_path: str, fps: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _, _, h, w = dataset.observations.shape
    
    scale = 4
    out_h, out_w = h * scale, w * scale
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for i in range(len(dataset.observations)):
        action_idx = np.argmax(dataset.actions[i])
        reward = dataset.rewards[i]
        term = dataset.terminations[i]

        frame = (((dataset.observations[i] + 1.0) / 2.0) * 255.0).astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        text = f"Act:{action_idx} Rew:{reward:.1f} Term:{term}"
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()

if __name__ == '__main__':
    start_idx = 0
    steps = 200
    video_fps = 15
    output_path = 'output/videos/rollout/rollout_video_1.mp4'
    sequence_length = 64
    latent_dim = 32
    codes_per_latent = 32
    epoch = 100
    env_name = "ALE/Pong-v5"
    weights_path = 'output/run/checkpoints/checkpoint_step_10000.pth'
    device = 'cuda'

    visualize_reconstruction(env_name=env_name, weights_path=weights_path, device=device, 
                             sequence_length=1000, latent_dim=latent_dim, codes_per_latent=codes_per_latent, 
                             epoch=epoch)