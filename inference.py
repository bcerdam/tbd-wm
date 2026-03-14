import torch
import argparse
import yaml
import numpy as np
import gymnasium as gym
import ale_py
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
from scripts.models.dynamics_modeling.dynamics_model_step import SymLogTwoHotLoss ###


def collect_steps(env_name:str, 
                  frameskip:int, 
                  noop_max:int, 
                  observation_height_width:int, 
                  episodic_life:bool, 
                  min_reward:float, 
                  max_reward:float,  
                  context_length:int, 
                  device:str, 
                  batch_size:int, 
                  actor:Actor, 
                  latent_dim:int, 
                  codes_per_latent:int,
                  timestep_idx:int, 
                  imagination_horizon:int, 
                  xlstm_dm:XLSTM_DM) -> Tuple:
    
    gym.register_envs(ale_py)
    env = gym.make(id=env_name, frameskip=1)
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
    tensor_action = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device) # a_(t)

    observation = reshape_observation(normalize_observation(observation=observation))
    observation_tensor = torch.from_numpy(observation).unsqueeze(0).unsqueeze(0).to(device=device) # o_t
    latent_t = encoder.forward(observations_batch=observation_tensor,
                                batch_size=1,
                                sequence_length=1,
                                latent_dim=latent_dim,
                                codes_per_latent=codes_per_latent)
    latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) # z_t

    token = tokenizer.forward(latents_sampled_batch=latent_t, actions_batch=tensor_action) # token_t -> (z_t, a_t)
    context_tokens = token

    _, _, _, features = xlstm_dm.forward(tokens_batch=context_tokens)
    features = features[:, -1:, :] # h_t -> (token_t)

    start_saving = False
    ctx_counter = 0
    i = 0

    all_observations, all_actions, all_rewards, all_terminations = [], [], [], []
    with torch.no_grad():
        while True:
            if i == timestep_idx:
                start_saving = True

            next_observation, next_reward, next_termination, next_truncated, info = env.step(action) # o_(t+2), r_(t+2), t_(t+2), a_(t+1)

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
            observation_tensor = torch.from_numpy(next_observation).unsqueeze(0).unsqueeze(0).to(device=device) # o_t+2
            latent_t = encoder.forward(observations_batch=observation_tensor,
                                        batch_size=1,
                                        sequence_length=1,
                                        latent_dim=latent_dim,
                                        codes_per_latent=codes_per_latent)
            latent_t = sample(latents_batch=latent_t, batch_size=1, sequence_length=1) # z_t+2

            env_state_vec = torch.cat([latent_t.view(1, 1, -1), features], dim=-1) # (z_t+2, h_t+1)

            action_logits = actor(state=env_state_vec)
            action_idx = torch.argmax(OneHotCategorical(logits=action_logits).sample()).item()
            action_array = np.zeros(env.action_space.n, dtype=np.float32)
            action_array[action_idx] = 1.0
            tensor_action = torch.from_numpy(action_array).unsqueeze(0).unsqueeze(0).to(device=device) # a_(t+2)

            observation = next_observation
            action = action_idx

            token = tokenizer.forward(latents_sampled_batch=latent_t, actions_batch=tensor_action) # token_t -> (z_t+2, a_t+2)

            context_tokens = torch.cat([context_tokens, token], dim=1)[:, -context_length:]
            _, _, _, features = xlstm_dm.forward(tokens_batch=context_tokens)
            features = features[:, -1:, :]

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
          actor:Actor) -> Tuple:
    
    symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20).to(tokens.device)
    
    context_length_limit = tokens.shape[1]

    with torch.no_grad():
        next_latents, rewards, terminations, all_features = xlstm_dm.forward(tokens_batch=tokens)

    latent_pred = next_latents[:, -1:, :]
    reward_pred = rewards[:, -1:, :]
    term_pred = terminations[:, -1:, :]
    h_t = all_features[:, -1:, :]
    
    imagined_frames = []
    imagined_actions = []
    imagined_rewards = []
    imagined_terminations = []
    features = []

    current_tokens = tokens

    for step in range(imagination_horizon):
        latent_pred = latent_pred.view(batch_size, 1, latent_dim, codes_per_latent)
        next_latent_sample = sample(latents_batch=latent_pred, batch_size=batch_size, sequence_length=1)

        decoded_reward = symlog_twohot_loss_func.decode(reward_pred[:, -1, :])
        imagined_rewards.append(decoded_reward)
        imagined_terminations.append((term_pred[:, -1, :] > 0.0).float())

        current_feature = h_t.squeeze(1)
        features.append(current_feature)

        next_latent_sample_flattened = next_latent_sample.view(batch_size, -1)
        env_state = torch.cat([next_latent_sample_flattened, current_feature], dim=-1).detach()

        action_logits = actor.forward(state=env_state)
        next_action_idx = torch.argmax(action_logits, dim=-1, keepdim=True)
        # next_action_idx = torch.argmax(OneHotCategorical(logits=action_logits).sample(), dim=-1, keepdim=True).item()
        next_action = torch.zeros_like(action_logits).scatter_(-1, next_action_idx, 1.0)
        # next_action = torch.zeros((batch_size, actor.env_actions), dtype=torch.float32, device=tokens.device)
        # next_action[:, 0] = 1.0
        # policy = OneHotCategorical(logits=action_logits)
        # next_action = policy.sample()

        imagined_actions.append(next_action)

        next_token = tokenizer.forward(latents_sampled_batch=next_latent_sample, actions_batch=next_action.unsqueeze(1))

        current_tokens = torch.cat([current_tokens, next_token], dim=1)
        if current_tokens.shape[1] > context_length_limit:
            current_tokens = current_tokens[:, 1:, :]

        with torch.no_grad():
            next_latents, rewards, terminations, all_features = xlstm_dm.forward(tokens_batch=current_tokens)

        latent_pred = next_latents[:, -1:, :]
        reward_pred = rewards[:, -1:, :]
        term_pred = terminations[:, -1:, :]
        h_t = all_features[:, -1:, :]

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

        imagined_frames, imagined_rewards, imagined_terminations, = dream(xlstm_dm=xlstm_dm, 
                                                                          decoder=decoder,
                                                                          tokenizer=tokenizer,
                                                                          tokens=tokens, 
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