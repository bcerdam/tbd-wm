import torch
import argparse
import yaml
import numpy as np
import gymnasium as gym
import ale_py
from scripts.utils.tensor_utils import normalize_observation, reshape_observation, env_n_actions
from scripts.utils.debug_utils import save_real_video
from gymnasium.wrappers import AtariPreprocessing, ClipReward
from typing import Tuple, List
from scripts.models.agent.actor import Actor
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.sampler import sample
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from torch.distributions import OneHotCategorical


def run_episode(env_name: str, 
                frameskip: int,
                noop_max: int,
                episodic_life: bool,
                min_reward: float,
                max_reward: float,
                observation_height_width: int, 
                actor:Actor, 
                encoder:CategoricalEncoder, 
                tokenizer:Tokenizer, 
                xlstm_dm:XLSTM_DM, 
                latent_dim:int, 
                codes_per_latent:int, 
                device:str, 
                context_length:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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
    termination = False
    truncated = False
    first_iter = True
    with torch.no_grad():
        while termination == False:
            all_observations.append(observation)

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
            all_actions.append(action_array)

            token = tokenizer.forward(latents_sampled_batch=sampled_latent, actions_batch=tensor_action_array)
            _, _, _, features, state = xlstm_dm.step(tokens_batch=token, state=state)

            observation, reward, termination, truncated, info = env.step(action)
            observation = reshape_observation(normalize_observation(observation=observation))

            all_rewards.append(reward)
            all_terminations.append(termination)

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


    encoder = CategoricalEncoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
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
    tokenizer.load_state_dict(clean_state_dict(checkpoint['tokenizer']))
    xlstm_dm.load_state_dict(clean_state_dict(checkpoint['dynamics']))
    actor.load_state_dict(clean_state_dict(checkpoint['actor']))

    encoder.eval()
    tokenizer.eval()
    xlstm_dm.eval()
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
                                                                               tokenizer=tokenizer, 
                                                                               xlstm_dm=xlstm_dm, 
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