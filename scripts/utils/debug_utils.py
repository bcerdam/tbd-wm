import h5py
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import sys
from typing import List, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.models.categorical_vae.sampler import sample
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder


def inspect_dataset(h5_path:str) -> None:
    with h5py.File(h5_path, 'r') as f:
        print(f"Inspecting: {h5_path}")
        for key in f.keys():
            dset = f[key]
            print(f"\nColumn: {key}")
            print(f"  Rows: {dset.shape[0]}")
            print(f"  Dtype: {dset.dtype}")
            print(f"  Min: {np.min(dset)}")
            print(f"  Max: {np.max(dset)}")


def rollout_video(h5_path:str, start_idx:int, steps:int, video_fps:int, output_path:str) -> None:
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(h5_path, 'r') as f:
        observations = f['observations'][start_idx : start_idx + steps]
        rollout_obs = np.moveaxis(observations, 1, -1)
        rollout_obs = ((rollout_obs + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        height, width = rollout_obs.shape[1], rollout_obs.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        for frame in rollout_obs:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()


def save_loss_history(new_losses: List[Dict[str, float]], output_dir: str) -> None: # Cluster
    keys = new_losses[0].keys()
    epoch_means = {}
    for k in keys:
        valid_vals = [d[k] for d in new_losses if d[k] is not None]
        epoch_means[k] = np.mean(valid_vals) if valid_vals else np.nan

    os.makedirs(output_dir, exist_ok=True) # Cluster
    history_path = os.path.join(output_dir, 'loss_history.npy') # Cluster

    loss_history = np.load(history_path, allow_pickle=True).item() if os.path.exists(history_path) else {}

    for k, v in epoch_means.items():
        loss_history.setdefault(k, []).append(v)

    np.save(history_path, loss_history)


def plot_current_loss(training_steps_per_epoch: int, epochs: int, output_dir: str) -> None: # Cluster
    history_path = os.path.join(output_dir, 'loss_history.npy') # Cluster
    loss_history = np.load(history_path, allow_pickle=True).item() # Cluster

    current_epoch = len(loss_history['total'])
    max_x = epochs * training_steps_per_epoch
    x_values = np.arange(1, current_epoch + 1) * training_steps_per_epoch

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), dpi=200, sharex=True)
    
    ax_wm = axes[0]
    wm_styles = {'total': {'color': '#D32F2F', 'label': 'Total'}}
    
    for key, style in wm_styles.items():
        if key in loss_history:
            ax_wm.plot(x_values, loss_history[key], linewidth=1.0, alpha=0.9, **style)
    
    ax_wm.set_title("World Model Loss", fontsize=7, fontweight='bold')
    ax_wm.set_ylabel("Loss", fontsize=6)
    ax_wm.legend(fontsize=5, loc='upper right', framealpha=0.8)
    ax_wm.grid(True, linestyle='--', alpha=0.3)

    ax_ac = axes[1]
    ac_styles = {
        'actor':  {'color': '#E91E63', 'label': 'Actor'},
        'critic': {'color': '#673AB7', 'label': 'Critic'},
    }

    for key, style in ac_styles.items():
        if key in loss_history:
            ax_ac.plot(x_values, loss_history[key], linewidth=1.0, alpha=0.9, **style)

    ax_ac.set_title("Actor Critic Loss", fontsize=7, fontweight='bold')
    ax_ac.set_ylabel("Loss", fontsize=6)
    ax_ac.legend(fontsize=5, loc='upper right', framealpha=0.8)
    ax_ac.grid(True, linestyle='--', alpha=0.3)

    ax_im = axes[2]
    if 'mean_episode_reward' in loss_history:
        y_vals = np.array(loss_history['mean_episode_reward'], dtype=float)
        mask = ~np.isnan(y_vals)
        ax_im.plot(x_values[mask], y_vals[mask], color='#FF9800', linewidth=1.0, marker='o', markersize=2, label='Mean Episode Reward')

    ax_im.set_title("Mean Episode Reward", fontsize=7, fontweight='bold')
    ax_im.set_ylabel("Reward", fontsize=6)
    ax_im.set_xlabel("Total Training Steps", fontsize=6)
    ax_im.legend(fontsize=5, loc='upper left', framealpha=0.8)
    ax_im.grid(True, linestyle='--', alpha=0.3)
    
    ax_im.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}K'))
    ax_im.set_xlim(0, max_x)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.jpeg'), format='jpeg', dpi=200, bbox_inches='tight') # Cluster
    plt.close()


def save_checkpoint(encoder, decoder, tokenizer, dynamics, 
                    actor, critic, ema_critic,
                    wm_optimizer, agent_optimizer, 
                    scaler, step, path="output/checkpoints"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'step': step,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'tokenizer': tokenizer.state_dict(),
        'dynamics': dynamics.state_dict(),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'ema_critic': ema_critic.state_dict(),
        'wm_optimizer': wm_optimizer.state_dict(),
        'agent_optimizer': agent_optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }, os.path.join(path, f"checkpoint_step_{step}.pth"))


import os
import cv2
import numpy as np
import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.categorical_vae.sampler import sample
from scripts.utils.tensor_utils import normalize_observation, reshape_observation

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

    orig_np = ((model_input[0].permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    recon_np = ((reconstructions[0].permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

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


# def save_dream_video(imagined_frames:List[np.ndarray], video_path:str, fps:int) -> None:
#     os.makedirs(os.path.dirname(video_path), exist_ok=True)

#     _, _, height, width = imagined_frames[0].shape

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

#     for frame in imagined_frames:
#         frame = frame[0]
#         frame = np.transpose(frame, (1, 2, 0))
#         frame = (frame + 1) * 127.5
#         frame = frame.astype(np.uint8)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         out.write(frame)

#     out.release()

import os
import cv2
import torch
import numpy as np
from typing import List, Union

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

import os
import cv2
import torch
import numpy as np
from typing import List, Union

def save_dream_video(real_frames: List[np.ndarray], # CHANGED
                     imagined_frames: List[np.ndarray], 
                     real_rewards: List[Union[torch.Tensor, np.ndarray]], # CHANGED
                     imagined_rewards: List[Union[torch.Tensor, np.ndarray]], # CHANGED
                     real_terminations: List[Union[torch.Tensor, np.ndarray]], # CHANGED
                     imagined_terminations: List[Union[torch.Tensor, np.ndarray]], # CHANGED
                     video_path: str, 
                     fps: int) -> None:
    
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    _, _, orig_height, orig_width = imagined_frames[0].shape
    scale = 4
    height, width = orig_height * scale, orig_width * scale
    double_width = width * 2 # CHANGED

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (double_width, height)) # CHANGED

    for r_f, i_f, r_rew, i_rew, r_term, i_term in zip(real_frames, imagined_frames, real_rewards, imagined_rewards, real_terminations, imagined_terminations): # CHANGED
        
        r_img = np.transpose(r_f[0], (1, 2, 0)) # CHANGED
        r_img = np.clip((r_img + 1) * 127.5, 0, 255).astype(np.uint8) # CHANGED
        r_img = cv2.cvtColor(r_img, cv2.COLOR_GRAY2BGR) if r_img.shape[-1] == 1 else cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR) # CHANGED
        r_img = cv2.resize(r_img, (width, height), interpolation=cv2.INTER_NEAREST) # CHANGED

        i_img = np.transpose(i_f[0], (1, 2, 0))
        i_img = np.clip((i_img + 1) * 127.5, 0, 255).astype(np.uint8)
        i_img = cv2.cvtColor(i_img, cv2.COLOR_GRAY2BGR) if i_img.shape[-1] == 1 else cv2.cvtColor(i_img, cv2.COLOR_RGB2BGR)
        i_img = cv2.resize(i_img, (width, height), interpolation=cv2.INTER_NEAREST)

        r_r_val = r_rew[0].item() if hasattr(r_rew[0], 'item') else float(r_rew[0]) # CHANGED
        r_t_val = r_term[0].item() if hasattr(r_term[0], 'item') else float(r_term[0]) # CHANGED
        
        i_r_val = i_rew[0].item() if hasattr(i_rew[0], 'item') else float(i_rew[0])
        i_t_val = i_term[0].item() if hasattr(i_term[0], 'item') else float(i_term[0])

        cv2.putText(r_img, f"Real R: {r_r_val:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA) # CHANGED
        cv2.putText(r_img, f"Real T: {r_t_val:.3f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # CHANGED

        cv2.putText(i_img, f"Imag R: {i_r_val:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(i_img, f"Imag T: {i_t_val:.3f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        combined_frame = np.concatenate((r_img, i_img), axis=1) # CHANGED
        out.write(combined_frame)

    out.release()


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
    weights_path = 'output/checkpoints/checkpoint_step_90000.pth'
    device = 'cuda'

    visualize_reconstruction(env_name=env_name, weights_path=weights_path, device=device, 
                             sequence_length=1000, latent_dim=latent_dim, codes_per_latent=codes_per_latent, 
                             epoch=epoch, video_path=output_path)

#     plot_current_loss(training_steps_per_epoch=200, epochs=500)

    # inspect_dataset(h5_path=h5_path)

    # rollout_video(h5_path=h5_path, 
    #               start_idx=start_idx, 
    #               steps=steps,
    #               video_fps=video_fps, 
    #               output_path=output_path)

    # visualize_reconstruction(dataset_path='data/replay_buffer.h5', 
    #                          weights_path='output/checkpoints/checkpoint_autoencoder_epoch_100.pth', 
    #                          device='cuda',
    #                          sequence_length=sequence_length, 
    #                          latent_dim=latent_dim, 
    #                          codes_per_latent=codes_per_latent, 
    #                          epoch=epoch)