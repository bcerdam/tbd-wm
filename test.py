import torch
import argparse
import yaml
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
from scripts.utils.debug_utils import save_real_video
from typing import Tuple, List
from scripts.models.agent.actor import Actor
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.sampler import sample
from scripts.models.dynamics_modeling.transformer_model import StochasticTransformerKVCache, DistHead
from scripts.models.dynamics_modeling.attention_blocks import get_subsequent_mask_with_batch_length
from torch.distributions import OneHotCategorical
from scripts.utils.tensor_utils import normalize_observation, reshape_observation, env_n_actions, MaxLast2FrameSkipWrapper, LifeLossInfo


def run_episode(env_name: str,
                frameskip: int,
                noop_max: int,
                episodic_life: bool,
                min_reward: float,
                max_reward: float,
                observation_height_width: int,
                actor: Actor,
                encoder: CategoricalEncoder,
                storm_transformer: StochasticTransformerKVCache,
                dist_head: DistHead,
                latent_dim: int,
                codes_per_latent: int,
                device: str,
                context_length: int) -> Tuple:
    """
    
    - context_obs deque(maxlen=16), context_action deque(maxlen=16)
    - Each step: encode full context, run full transformer, get prior + hidden, select action
    - No KV cache (correct positional encoding every step)
    """
    gym.register_envs(ale_py)
    env = gym.make(id=env_name, frameskip=1, full_action_space=False, render_mode="rgb_array")
    env = MaxLast2FrameSkipWrapper(env, skip=frameskip)
    env = gym.wrappers.ResizeObservation(env, shape=(observation_height_width, observation_height_width))
    env = LifeLossInfo(env)

    # Context deques with maxlen=16 (matches STORM eval.py line 60-61)
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminations = []

    observation, info = env.reset()

    termination = False
    truncated = False

    with torch.no_grad():
        while not (termination or truncated):
            # === Select action using full context ===
            if len(context_action) == 0:
                action = env.action_space.sample()
            else:
                obs_context = torch.cat(list(context_obs), dim=1)  # (1, T, C, H, W)
                T = obs_context.shape[1]

                latents = encoder.forward(observations_batch=obs_context,
                                          batch_size=1,
                                          sequence_length=T,
                                          latent_dim=latent_dim,
                                          codes_per_latent=codes_per_latent)
                latents_sampled = sample(latents_batch=latents, batch_size=1, sequence_length=T)
                flattened = latents_sampled.flatten(start_dim=2)

                act_context = torch.tensor(list(context_action), device=device).unsqueeze(0)

                temporal_mask = get_subsequent_mask_with_batch_length(T, device)
                dist_feat = storm_transformer(flattened, act_context, temporal_mask)
                last_dist_feat = dist_feat[:, -1:]

                prior_logits = dist_head.forward_prior(last_dist_feat)
                prior_sample = OneHotCategorical(logits=prior_logits).sample()
                prior_flat = prior_sample.flatten(start_dim=2)

                env_state = torch.cat([prior_flat, last_dist_feat], dim=-1)
                action_logits = actor(state=env_state)
                action = torch.argmax(OneHotCategorical(logits=action_logits).sample()).item()

            # === Append to context BEFORE stepping ===
            obs_normalized = reshape_observation(normalize_observation(observation))
            obs_tensor = torch.from_numpy(obs_normalized).unsqueeze(0).unsqueeze(0).to(device=device)
            context_obs.append(obs_tensor)
            context_action.append(action)

            # === Store for output ===
            all_observations.append(obs_normalized)
            action_array = np.zeros(env.action_space.n, dtype=np.float32)
            action_array[action] = 1.0
            all_actions.append(action_array)

            # === Step environment ===
            next_observation, next_reward, termination, truncated, info = env.step(action)
            all_rewards.append(next_reward)
            all_terminations.append(termination)

            observation = next_observation

    all_observations = np.array(all_observations)
    all_actions = np.array(all_actions)
    all_rewards = np.array(all_rewards)
    all_terminations = np.array(all_terminations)

    return all_observations, all_actions, all_rewards, all_terminations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', default='config/train_wm.yaml', type=str, help='Path to env parameters .yaml file')
    parser.add_argument('--env_cfg', default='config/env.yaml', type=str, help='Path to env parameters .yaml file')
    parser.add_argument('--inference_cfg', default='config/inference.yaml', type=str, help='Path to inference hyperparameters .yaml file')
    args = parser.parse_args()

    with open(args.train_cfg, 'r') as file_train, open(args.env_cfg, 'r') as file_env, open(args.inference_cfg, 'r') as file_inference:
        train_cfg = yaml.safe_load(file_train)['train_wm']
        inference_cfg = yaml.safe_load(file_inference)['inference']
        env_cfg = yaml.safe_load(file_env)['env']

    ENV_NAME = env_cfg['env_name']
    WEIGHTS_PATH = inference_cfg['weights_path']
    VIDEO_PATH = f'output/videos/dream/dream_{ENV_NAME}.mp4'
    FPS = inference_cfg['fps']
    BATCH_SIZE = inference_cfg['batch_size']
    CONTEXT_LENGTH = inference_cfg['context_length']
    IMAGINATION_HORIZON = inference_cfg['imagination_horizon']
    LATENT_DIM = train_cfg['latent_dim']
    CODES_PER_LATENT = train_cfg['codes_per_latent']
    DEVICE = 'cuda'
    ENV_ACTIONS = env_n_actions(env_name=ENV_NAME)
    EMBEDDING_DIM = train_cfg['embedding_dim']
    SEQUENCE_LENGTH = train_cfg['sequence_length']
    NUM_BLOCKS = train_cfg['num_blocks']
    DROPOUT = train_cfg['dropout']
    NUM_HEADS = train_cfg['num_heads']
    FRAMESKIP = env_cfg['frameskip']
    NOOP_MAX = env_cfg['noop_max']
    OBSERVATION_HEIGHT_WIDTH = env_cfg['observation_height_width']
    EPISODIC_LIFE = env_cfg['episodic_life']
    MIN_REWARD = env_cfg['min_reward']
    MAX_REWARD = env_cfg['max_reward']

    encoder = CategoricalEncoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    storm_transformer = StochasticTransformerKVCache(stoch_dim=LATENT_DIM*LATENT_DIM,
                                                     action_dim=ENV_ACTIONS,
                                                     feat_dim=EMBEDDING_DIM,
                                                     num_layers=NUM_BLOCKS,
                                                     num_heads=NUM_HEADS,
                                                     max_length=SEQUENCE_LENGTH,
                                                     dropout=DROPOUT).to(DEVICE)
    dist_head = DistHead(image_feat_dim=0, transformer_hidden_dim=EMBEDDING_DIM, stoch_dim=LATENT_DIM).to(DEVICE)

    actor = Actor(latent_dim=LATENT_DIM,
                  codes_per_latent=CODES_PER_LATENT,
                  embedding_dim=EMBEDDING_DIM,
                  env_actions=ENV_ACTIONS).to(DEVICE)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    def clean_state_dict(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    encoder.load_state_dict(clean_state_dict(checkpoint['encoder']))
    storm_transformer.load_state_dict(clean_state_dict(checkpoint['storm_transformer']))
    dist_head.load_state_dict(clean_state_dict(checkpoint['dist_head']))
    actor.load_state_dict(clean_state_dict(checkpoint['actor']))

    encoder.eval()
    storm_transformer.eval()
    dist_head.eval()
    actor.eval()

    all_observations, all_actions, all_rewards, all_terminations = run_episode(env_name=ENV_NAME,
                                                                               frameskip=FRAMESKIP,
                                                                               noop_max=NOOP_MAX,
                                                                               episodic_life=EPISODIC_LIFE,
                                                                               min_reward=MIN_REWARD,
                                                                               max_reward=MAX_REWARD,
                                                                               observation_height_width=OBSERVATION_HEIGHT_WIDTH,
                                                                               actor=actor,
                                                                               encoder=encoder,
                                                                               storm_transformer=storm_transformer,
                                                                               dist_head=dist_head,
                                                                               latent_dim=LATENT_DIM,
                                                                               codes_per_latent=CODES_PER_LATENT,
                                                                               device=DEVICE,
                                                                               context_length=CONTEXT_LENGTH)

    print(all_rewards)
    print(f'Sum rewards: {np.sum(all_rewards)}')

    all_observations = np.expand_dims(all_observations, axis=1)
    all_rewards = np.expand_dims(all_rewards, axis=1)
    all_terminations = np.expand_dims(all_terminations, axis=1)

    save_real_video(imagined_frames=all_observations,
                     imagined_rewards=all_rewards,
                     imagined_terminations=all_terminations,
                     video_path=VIDEO_PATH,
                     fps=FPS)