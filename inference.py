import torch
import argparse
import yaml
import numpy as np
import gymnasium as gym
import ale_py
import cv2
from scripts.utils.tensor_utils import normalize_observation, reshape_observation
from gymnasium.wrappers import AtariPreprocessing, ClipReward
from typing import List, Tuple
from scripts.utils.debug_utils import save_dream_video
from scripts.utils.tensor_utils import env_n_actions
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.categorical_vae.sampler import sample
from scripts.models.agent.actor import Actor
from torch.distributions import OneHotCategorical


def collect_steps(env_name:str, 
                  frameskip:int, 
                  noop_max:int, 
                  observation_height_width:int, 
                  episodic_life:bool, 
                  min_reward:float, 
                  max_reward:float,  
                  context_length:int, 
                  env_actions:int, 
                  device:str, 
                  batch_size:int, 
                  actor:Actor, 
                  latent_dim:int, 
                  codes_per_latent:int,
                  timestep_idx:int, 
                  imagination_horizon:int, 
                  xlstm_dm:XLSTM_DM) -> Tuple[torch.Tensor, torch.Tensor]:
    
    gym.register_envs(ale_py)
    env = gym.make(id=env_name, frameskip=1)
    env = AtariPreprocessing(env=env, 
                             noop_max=noop_max, 
                             frame_skip=frameskip, 
                             screen_size=observation_height_width, 
                             terminal_on_life_loss=episodic_life, 
                             grayscale_obs=False)
    env = ClipReward(env=env, min_reward=min_reward, max_reward=max_reward)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminations = [] 
    all_episode_starts = []

    state = None
    embedding_dim = tokenizer.embedding_dim 
    features = torch.zeros(1, 1, embedding_dim, device=device)

    observation, info = env.reset()
    observation = reshape_observation(normalize_observation(observation=observation))
    episode_start = True
    first_iter = True
    start_saving = False
    ctx_counter = 0
    i = 0
    termination = False
    with torch.no_grad():
        while termination == False:
            if i == timestep_idx:
                start_saving = True

            if start_saving == True:
                all_observations.append(observation)
                ctx_counter += 1
            
            observation_tensor = torch.from_numpy(observation).unsqueeze(0).unsqueeze(0).to(device=device)
            latent = encoder.forward(observations_batch=observation_tensor, 
                                    batch_size=1, 
                                    sequence_length=1, 
                                    latent_dim=latent_dim, 
                                    codes_per_latent=codes_per_latent)
            sampled_latent = sample(latents_batch=latent, batch_size=1, sequence_length=1)

            action_array = np.zeros(env.action_space.n, dtype=np.float32)
            env_state = torch.concat([sampled_latent, features], dim=-1)


            if first_iter == True:
                action = env.action_space.sample()
                first_iter = False
            else:
                action_logits = actor(state=env_state)
                policy = OneHotCategorical(logits=action_logits)
                action = torch.argmax(policy.sample()).item()

            action_array[action] = 1.0
            tensor_action_array = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device)
            if start_saving == True:
                all_actions.append(action_array)

            token = tokenizer.forward(latents_sampled_batch=sampled_latent, actions_batch=tensor_action_array)
            _, _, _, features, state = xlstm_dm.step(tokens_batch=token, state=state)

            observation, reward, termination, truncated, info = env.step(action)
            observation = reshape_observation(normalize_observation(observation=observation))

            if start_saving == True:
                all_rewards.append(reward)
                all_terminations.append(termination)

        
            if termination or truncated:
                observation, info = env.reset()

            if ctx_counter == (context_length+imagination_horizon):
                break

            i += 1
            
    env.close()

    observations = torch.from_numpy(np.stack(all_observations)).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
    actions = torch.from_numpy(np.stack(all_actions)).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    rewards = torch.from_numpy(np.stack(all_rewards)).unsqueeze(0).repeat(batch_size, 1).to(device)
    terminations = torch.from_numpy(np.stack(all_terminations)).unsqueeze(0).repeat(batch_size, 1).to(device)

    return observations, actions, rewards, terminations


def dream(xlstm_dm:XLSTM_DM, 
          decoder:CategoricalDecoder, 
          tokenizer:Tokenizer,
          tokens:torch.Tensor, 
          imagination_horizon:int, 
          latent_dim:int, 
          codes_per_latent:int, 
          batch_size:int, 
          env_actions:int,
          device:str, 
          actor:Actor) -> Tuple:
    
    state = None
    context_length = tokens.shape[1]
    for t in range(context_length):
        batch_tokens_t = tokens[:, t:t+1, :]
        latent, reward, termination, feature, state = xlstm_dm.step(tokens_batch=batch_tokens_t, state=state)
    
    imagined_frames = []
    imagined_actions = []
    imagined_latents = []
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


        next_latent_sample = sample(latents_batch=next_latent, batch_size=batch_size, sequence_length=1)

        decoded_latent = decoder.forward(latents_batch=next_latent_sample, 
                                        batch_size=batch_size, 
                                        sequence_length=1, 
                                        latent_dim=latent_dim, 
                                        codes_per_latent=codes_per_latent).squeeze(1).cpu().numpy()
        
        imagined_frames.append(decoded_latent)

    return imagined_frames, imagined_latents, imagined_rewards, imagined_terminations, features


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
    SLSTM_AT = train_cfg['slstm_at']
    DROPOUT = train_cfg['dropout']
    ADD_POST_BLOCKS_NORM = train_cfg['add_post_blocks_norm']
    CONV1D_KERNEL_SIZE = train_cfg['conv1d_kernel_size']
    NUM_HEADS = train_cfg['num_heads']
    QKV_PROJ_BLOCKSIZE = train_cfg['qkv_proj_blocksize']
    BIAS_INIT = train_cfg['bias_init']
    PROJ_FACTOR = train_cfg['proj_factor']
    ACT_FN = train_cfg['act_fn']
    FRAMESKIP = env_cfg['frameskip']
    NOOP_MAX = env_cfg['noop_max']
    OBSERVATION_HEIGHT_WIDTH = env_cfg['observation_height_width']
    EPISODIC_LIFE = env_cfg['episodic_life']
    MIN_REWARD = env_cfg['min_reward']
    MAX_REWARD = env_cfg['max_reward']


    dataset = AtariDataset(sequence_length=CONTEXT_LENGTH)
    encoder = CategoricalEncoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    decoder = CategoricalDecoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    tokenizer = Tokenizer(latent_dim=LATENT_DIM, 
                          codes_per_latent=CODES_PER_LATENT, 
                          env_actions=ENV_ACTIONS, 
                          embedding_dim=EMBEDDING_DIM, 
                          sequence_length=SEQUENCE_LENGTH).to(DEVICE)
    xlstm_dm = XLSTM_DM(sequence_length=SEQUENCE_LENGTH, 
                        num_blocks=NUM_BLOCKS, 
                        embedding_dim=EMBEDDING_DIM, 
                        slstm_at=SLSTM_AT, 
                        dropout=DROPOUT, 
                        add_post_blocks_norm=ADD_POST_BLOCKS_NORM, 
                        conv1d_kernel_size=CONV1D_KERNEL_SIZE, 
                        qkv_proj_blocksize=QKV_PROJ_BLOCKSIZE, 
                        num_heads=NUM_HEADS, 
                        latent_dim=LATENT_DIM, 
                        codes_per_latent=CODES_PER_LATENT, 
                        bias_init=BIAS_INIT, 
                        proj_factor=PROJ_FACTOR, 
                        act_fn=ACT_FN).to(DEVICE)

    actor = Actor(latent_dim=LATENT_DIM, 
                codes_per_latent=CODES_PER_LATENT, 
                embedding_dim=EMBEDDING_DIM, 
                env_actions=ENV_ACTIONS).to(DEVICE)
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    def clean_state_dict(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    encoder.load_state_dict(clean_state_dict(checkpoint['encoder']))
    decoder.load_state_dict(clean_state_dict(checkpoint['decoder']))
    tokenizer.load_state_dict(clean_state_dict(checkpoint['tokenizer']))
    xlstm_dm.load_state_dict(clean_state_dict(checkpoint['dynamics']))
    actor.load_state_dict(clean_state_dict(checkpoint['actor']))

    encoder.eval()
    decoder.eval()
    tokenizer.eval()
    xlstm_dm.eval()
    actor.eval()
    
    observations, actions, rewards, terminations = collect_steps(env_name=ENV_NAME, 
                                                                frameskip=FRAMESKIP, 
                                                                noop_max=NOOP_MAX, 
                                                                observation_height_width=OBSERVATION_HEIGHT_WIDTH, 
                                                                episodic_life=EPISODIC_LIFE, 
                                                                min_reward=MIN_REWARD, 
                                                                max_reward=MAX_REWARD, 
                                                                context_length=CONTEXT_LENGTH, 
                                                                env_actions=ENV_ACTIONS, 
                                                                device=DEVICE, 
                                                                batch_size=BATCH_SIZE, 
                                                                actor=actor, 
                                                                latent_dim=LATENT_DIM, 
                                                                codes_per_latent=CODES_PER_LATENT, 
                                                                imagination_horizon=IMAGINATION_HORIZON, 
                                                                timestep_idx=inference_cfg['timestep_idx'], 
                                                                xlstm_dm=xlstm_dm)

    with torch.no_grad():
        latents = encoder.forward(observations_batch=observations[:, :CONTEXT_LENGTH], 
                                  batch_size=BATCH_SIZE, 
                                  sequence_length=CONTEXT_LENGTH, 
                                  latent_dim=LATENT_DIM, 
                                  codes_per_latent=CODES_PER_LATENT)
        
        latents_sampled_batch = sample(latents, batch_size=BATCH_SIZE, sequence_length=CONTEXT_LENGTH)

        tokens = tokenizer.forward(latents_sampled_batch=latents_sampled_batch, actions_batch=actions[:, :CONTEXT_LENGTH])

        imagined_frames, _, imagined_rewards, imagined_terminations, _ = dream(xlstm_dm=xlstm_dm, 
                                                                               decoder=decoder,
                                                                               tokenizer=tokenizer,
                                                                               tokens=tokens, 
                                                                               imagination_horizon=IMAGINATION_HORIZON, 
                                                                               latent_dim=LATENT_DIM, 
                                                                               codes_per_latent=CODES_PER_LATENT, 
                                                                               batch_size=BATCH_SIZE, 
                                                                               env_actions=ENV_ACTIONS, 
                                                                               device=DEVICE, 
                                                                               actor=actor)
    
        
        save_dream_video(real_frames=[f.numpy() for f in torch.unbind(observations[:, CONTEXT_LENGTH:].cpu(), dim=1)],
                         imagined_frames=imagined_frames, 
                         real_rewards=torch.unbind(rewards[:, CONTEXT_LENGTH:].cpu(), dim=1),
                         imagined_rewards=imagined_rewards,
                         real_terminations=torch.unbind(terminations[:, CONTEXT_LENGTH:].cpu(), dim=1),
                         imagined_terminations=imagined_terminations, 
                         video_path=VIDEO_PATH, 
                         fps=FPS)