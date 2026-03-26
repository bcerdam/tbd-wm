import torch
import argparse
import yaml
import numpy as np
import gymnasium as gym
import ale_py
from scripts.utils.tensor_utils import normalize_observation, reshape_observation, FireOnLifeLossWrapper
from gymnasium.wrappers import AtariPreprocessing, ClipReward
from typing import List, Tuple
from scripts.utils.debug_utils import save_dream_video
from scripts.utils.tensor_utils import env_n_actions
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.dynamics_modeling.transformer_model import StochasticTransformerKVCache, DistHead, RewardDecoder, TerminationDecoder
from scripts.models.categorical_vae.sampler import sample
from scripts.models.agent.actor import Actor
from torch.distributions import OneHotCategorical
from scripts.models.dynamics_modeling.dynamics_model_step import SymLogTwoHotLoss


def collect_steps(env_name: str, 
                  frameskip: int, 
                  noop_max: int, 
                  observation_height_width: int, 
                  episodic_life: bool, 
                  min_reward: float, 
                  max_reward: float,  
                  context_length: int, 
                  device: str, 
                  batch_size: int, 
                  actor: Actor, 
                  encoder: CategoricalEncoder,
                  latent_dim: int, 
                  codes_per_latent: int,
                  timestep_idx: int, 
                  imagination_horizon: int, 
                  sequence_length: int,
                  storm_transformer: StochasticTransformerKVCache) -> Tuple:
    
    gym.register_envs(ale_py)
    env = gym.make(id=env_name, frameskip=1)
    env = FireOnLifeLossWrapper(env)
    env = AtariPreprocessing(env=env, 
                             noop_max=noop_max, 
                             frame_skip=frameskip, 
                             screen_size=observation_height_width, 
                             terminal_on_life_loss=episodic_life, 
                             grayscale_obs=False)
    env = ClipReward(env=env, min_reward=min_reward, max_reward=max_reward)

    observation, info = env.reset()
    lives = info.get("lives", 0)

    action = env.action_space.sample()
    action_array = np.zeros(env.action_space.n, dtype=np.float32)
    action_array[action] = 1.0

    observation = reshape_observation(normalize_observation(observation=observation))
    observation_tensor = torch.from_numpy(observation).unsqueeze(0).unsqueeze(0).to(device=device) 
    latent_t = encoder.forward(observations_batch=observation_tensor,
                                batch_size=1,
                                sequence_length=1,
                                latent_dim=latent_dim,
                                codes_per_latent=codes_per_latent)
    latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) 

    storm_transformer.reset_kv_cache_list(1, dtype=torch.bfloat16)

    flattened_sample = latent_t.flatten(start_dim=2)
    action_tensor_idx = torch.tensor([[action]], device=device)
    features = storm_transformer.forward_with_kv_cache(samples=flattened_sample, action=action_tensor_idx)

    start_saving = False
    ctx_counter = 0
    i = 0

    all_observations, all_actions, all_rewards, all_terminations = [], [], [], []
    with torch.no_grad():
        while True:
            if i == timestep_idx:
                start_saving = True

            next_observation, next_reward, next_termination, next_truncated, info = env.step(action) 

            current_lives = info.get("lives", 0)
            life_loss = current_lives < lives
            lives = current_lives
            done = next_termination or life_loss

            if start_saving == True:
                all_observations.append(observation)
                all_actions.append(action_array)
                all_rewards.append(next_reward)
                all_terminations.append(done)
                ctx_counter += 1
                
            next_observation = reshape_observation(normalize_observation(observation=next_observation))
            observation_tensor = torch.from_numpy(next_observation).unsqueeze(0).unsqueeze(0).to(device=device) 
            latent_t = encoder.forward(observations_batch=observation_tensor,
                                        batch_size=1,
                                        sequence_length=1,
                                        latent_dim=latent_dim,
                                        codes_per_latent=codes_per_latent)
            latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) 

            env_state_vec = torch.cat([latent_t.view(1, 1, -1), features], dim=-1) 

            action_logits = actor(state=env_state_vec)
            action_idx = torch.argmax(OneHotCategorical(logits=action_logits).sample()).item()
            action_array = np.zeros(env.action_space.n, dtype=np.float32)
            action_array[action_idx] = 1.0

            observation = next_observation
            action = action_idx

            if storm_transformer.kv_cache_list[0].shape[1] == sequence_length:
                for idx in range(len(storm_transformer.kv_cache_list)):
                    storm_transformer.kv_cache_list[idx] = storm_transformer.kv_cache_list[idx][:, 1:, :]

            flattened_sample = latent_t.flatten(start_dim=2)
            action_tensor_idx = torch.tensor([[action_idx]], device=device)
            features = storm_transformer.forward_with_kv_cache(samples=flattened_sample, action=action_tensor_idx)

            if ctx_counter == (context_length+imagination_horizon):
                break

            i += 1
            
    env.close()

    observations = torch.from_numpy(np.stack(all_observations)).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
    actions = torch.from_numpy(np.stack(all_actions)).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    rewards = torch.from_numpy(np.stack(all_rewards)).unsqueeze(0).repeat(batch_size, 1).to(device)
    terminations = torch.from_numpy(np.stack(all_terminations)).unsqueeze(0).repeat(batch_size, 1).to(device)

    state = {}
    return observations, actions, rewards, terminations, features, state


def dream(storm_transformer: StochasticTransformerKVCache,
          dist_head: DistHead,
          reward_decoder: RewardDecoder,
          termination_decoder: TerminationDecoder, 
          decoder: CategoricalDecoder, 
          latents_sampled_batch: torch.Tensor,
          actions_indices: torch.Tensor,
          imagination_horizon: int, 
          latent_dim: int, 
          codes_per_latent: int, 
          batch_size: int, 
          actor: Actor) -> Tuple:
    
    symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20).to(latents_sampled_batch.device)

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
    
    imagined_frames = []
    imagined_actions = []
    imagined_rewards = []
    imagined_terminations = []
    features = []

    for step in range(imagination_horizon):
        next_latent_sample = sample(latents_batch=latent_pred, batch_size=batch_size, sequence_length=1)

        with torch.no_grad():
            reward_pred = reward_decoder(dist_feat)
            term_pred = termination_decoder(dist_feat)

        decoded_reward = symlog_twohot_loss_func.decode(reward_pred[:, -1, :])
        imagined_rewards.append(decoded_reward)
        imagined_terminations.append((term_pred[:, -1] > 0.0).float())

        current_feature = h_t.squeeze(1)
        features.append(current_feature)

        next_latent_sample_flattened = next_latent_sample.flatten(start_dim=2)
        env_state = torch.cat([next_latent_sample_flattened.squeeze(1), current_feature], dim=-1).detach()

        action_logits = actor.forward(state=env_state)
        policy = OneHotCategorical(logits=action_logits)
        
        next_action_onehot = policy.sample()
        next_action_idx = torch.argmax(next_action_onehot, dim=-1).unsqueeze(1) 

        imagined_actions.append(next_action_idx)

        with torch.no_grad():
            dist_feat = storm_transformer.forward_with_kv_cache(
                samples=next_latent_sample_flattened, 
                action=next_action_idx
            )
            prior_logits = dist_head.forward_prior(dist_feat)

        latent_pred = prior_logits
        h_t = dist_feat

        with torch.no_grad():
            decoded_latent = decoder.forward(latents_batch=next_latent_sample,
                                             batch_size=batch_size,
                                             sequence_length=1,
                                             latent_dim=latent_dim,
                                             codes_per_latent=codes_per_latent).squeeze(1).cpu().numpy()

        imagined_frames.append(decoded_latent)

    return imagined_frames, imagined_rewards, imagined_terminations


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

    dataset = AtariDataset(sequence_length=CONTEXT_LENGTH)
    encoder = CategoricalEncoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    decoder = CategoricalDecoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    
    storm_transformer = StochasticTransformerKVCache(stoch_dim=LATENT_DIM*LATENT_DIM, 
                                                     action_dim=ENV_ACTIONS, 
                                                     feat_dim=EMBEDDING_DIM, 
                                                     num_layers=NUM_BLOCKS, 
                                                     num_heads=NUM_HEADS, 
                                                     max_length=SEQUENCE_LENGTH, 
                                                     dropout=DROPOUT).to(DEVICE)
    
    dist_head = DistHead(image_feat_dim=0, transformer_hidden_dim=EMBEDDING_DIM, stoch_dim=LATENT_DIM).to(DEVICE)
    reward_decoder = RewardDecoder(num_classes=255, embedding_size=LATENT_DIM*LATENT_DIM, transformer_hidden_dim=EMBEDDING_DIM).to(DEVICE)
    termination_decoder = TerminationDecoder(embedding_size=LATENT_DIM*LATENT_DIM, transformer_hidden_dim=EMBEDDING_DIM).to(DEVICE)

    actor = Actor(latent_dim=LATENT_DIM, 
                codes_per_latent=CODES_PER_LATENT, 
                embedding_dim=EMBEDDING_DIM, 
                env_actions=ENV_ACTIONS).to(DEVICE)
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    def clean_state_dict(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    encoder.load_state_dict(clean_state_dict(checkpoint['encoder']))
    decoder.load_state_dict(clean_state_dict(checkpoint['decoder']))
    storm_transformer.load_state_dict(clean_state_dict(checkpoint['storm_transformer']))
    dist_head.load_state_dict(clean_state_dict(checkpoint['dist_head']))
    reward_decoder.load_state_dict(clean_state_dict(checkpoint['reward_decoder']))
    termination_decoder.load_state_dict(clean_state_dict(checkpoint['termination_decoder']))
    actor.load_state_dict(clean_state_dict(checkpoint['actor']))

    encoder.eval()
    decoder.eval()
    storm_transformer.eval()
    dist_head.eval()
    reward_decoder.eval()
    termination_decoder.eval()
    actor.eval()
    
    observations, actions, rewards, terminations, features, state = collect_steps(env_name=ENV_NAME, 
                                                                                frameskip=FRAMESKIP, 
                                                                                noop_max=NOOP_MAX, 
                                                                                observation_height_width=OBSERVATION_HEIGHT_WIDTH, 
                                                                                episodic_life=EPISODIC_LIFE, 
                                                                                min_reward=MIN_REWARD, 
                                                                                max_reward=MAX_REWARD, 
                                                                                context_length=CONTEXT_LENGTH, 
                                                                                device=DEVICE, 
                                                                                batch_size=BATCH_SIZE, 
                                                                                actor=actor, 
                                                                                encoder=encoder,
                                                                                latent_dim=LATENT_DIM, 
                                                                                codes_per_latent=CODES_PER_LATENT, 
                                                                                imagination_horizon=IMAGINATION_HORIZON, 
                                                                                timestep_idx=inference_cfg['timestep_idx'], 
                                                                                sequence_length=SEQUENCE_LENGTH,
                                                                                storm_transformer=storm_transformer)

    with torch.no_grad():
        latents = encoder.forward(observations_batch=observations[:, :CONTEXT_LENGTH], 
                                  batch_size=BATCH_SIZE, 
                                  sequence_length=CONTEXT_LENGTH, 
                                  latent_dim=LATENT_DIM, 
                                  codes_per_latent=CODES_PER_LATENT)
        
        latents_sampled_batch = sample(latents, batch_size=BATCH_SIZE, sequence_length=CONTEXT_LENGTH)

        actions_batch = actions[:, :CONTEXT_LENGTH]
        if actions_batch.dim() == 3:
            actions_indices = torch.argmax(actions_batch, dim=-1)
        else:
            actions_indices = actions_batch

        imagined_frames, imagined_rewards, imagined_terminations, = dream(storm_transformer=storm_transformer,
                                                                          dist_head=dist_head,
                                                                          reward_decoder=reward_decoder,
                                                                          termination_decoder=termination_decoder,
                                                                          decoder=decoder,
                                                                          latents_sampled_batch=latents_sampled_batch,
                                                                          actions_indices=actions_indices,
                                                                          imagination_horizon=IMAGINATION_HORIZON, 
                                                                          latent_dim=LATENT_DIM, 
                                                                          codes_per_latent=CODES_PER_LATENT, 
                                                                          batch_size=BATCH_SIZE, 
                                                                          actor=actor)
    
        
        save_dream_video(real_frames=[f.numpy() for f in torch.unbind(observations[:, CONTEXT_LENGTH:].cpu(), dim=1)],
                         imagined_frames=imagined_frames, 
                         real_rewards=torch.unbind(rewards[:, CONTEXT_LENGTH:].cpu(), dim=1),
                         imagined_rewards=imagined_rewards,
                         real_terminations=torch.unbind(terminations[:, CONTEXT_LENGTH:].cpu(), dim=1),
                         imagined_terminations=imagined_terminations, 
                         video_path=VIDEO_PATH, 
                         fps=FPS)