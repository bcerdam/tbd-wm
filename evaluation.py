import argparse
import yaml
import torch
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
from torch.distributions import OneHotCategorical
from scripts.utils.tensor_utils import reshape_observation, MaxLast2FrameSkipWrapper, LifeLossInfo
from scripts.models.world_model.categorical_autoencoder.encoder import CategoricalEncoder
from scripts.models.world_model.categorical_autoencoder.sampler import sample
from scripts.models.world_model.transformer.latent_action_embedder import LatentActionEmbedder
from scripts.models.world_model.transformer.transformer import TransformerDecoder
from scripts.models.agent.train_agent import take_action
from scripts.models.agent.actor import Actor
from scripts.utils.tensor_utils import env_n_actions


def run_episode(env_name:str, 
                frameskip:int, 
                observation_height_width:int, 
                actor: Actor,
                categorical_encoder: CategoricalEncoder,
                latent_action_embedder: LatentActionEmbedder,
                transformer: TransformerDecoder,
                latent_dim: int,
                codes_per_latent: int,
                context_length: int,
                device: str,
                dtype: torch.dtype) -> float:

    gym.register_envs(ale_py)
    env = gym.make(id=env_name, frameskip=1, full_action_space=False, render_mode="rgb_array")
    env = MaxLast2FrameSkipWrapper(env, skip=frameskip)
    env = gym.wrappers.ResizeObservation(env, shape=(observation_height_width, observation_height_width))
    env = LifeLossInfo(env)

    env_actions = env.action_space.n

    observation, info = env.reset()
    observation = reshape_observation(observation)
    action = env.action_space.sample()

    total_reward = 0.0
    terminated = False
    truncated = False
    context_obs = deque(maxlen=context_length)
    context_act = deque(maxlen=context_length)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            while not (terminated or truncated):
                context_obs.append(observation)
                context_act.append(action)

                action = take_action(context_obs=context_obs, 
                                     context_act=context_act, 
                                     categorical_encoder=categorical_encoder, 
                                     latent_action_embedder=latent_action_embedder, 
                                     transformer=transformer, 
                                     actor=actor, 
                                     latent_dim=latent_dim, 
                                     codes_per_latent=codes_per_latent, 
                                     env_actions=env_actions, 
                                     device=device, 
                                     tensor_dtype=dtype)

                # obs_tensor = torch.from_numpy(np.stack(list(context_obs))).unsqueeze(0).to(device).to(dtype) / 255.0
                # act_tensor = torch.from_numpy(np.stack(list(context_act))).unsqueeze(0).to(device).float()
                # ctx_len = obs_tensor.shape[1]

                # posterior_raw_logits = encoder.forward(observations_batch=obs_tensor, 
                #                                     batch_size=1, 
                #                                     sequence_length=ctx_len,
                #                                     latent_dim=latent_dim,codes_per_latent=codes_per_latent)
                # posterior_sample, _ = sample(posterior_raw_logits=posterior_raw_logits)

                # embeddings = latent_action_embedder.forward(posterior_sample_batch=posterior_sample, actions_batch=act_tensor, position_offset=0)

                # mask = torch.tril(torch.ones((ctx_len, ctx_len), device=device)).unsqueeze(0).unsqueeze(0)
                # prior_raw_logits, _, _, x, _ = transformer.forward(x=embeddings, mask=mask)

                # prior_raw_logits = prior_raw_logits[:, -1:, :]
                # x = x[:, -1:, :]

                # prior_sample, _ = sample(posterior_raw_logits=prior_raw_logits.view(1, 1, latent_dim, codes_per_latent))
                # env_state = torch.cat([prior_sample.view(1, -1), x.squeeze(1)], dim=-1)

                # action_logits = actor.forward(state=env_state)
                # policy = OneHotCategorical(logits=action_logits)
                # action = torch.argmax(policy.sample()).item()

                next_observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                observation = reshape_observation(next_observation)

    env.close()
    return total_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_wm_cfg', default='config/train_wm.yaml', type=str)
    parser.add_argument('--train_agent_cfg', default='config/train_agent.yaml', type=str)
    parser.add_argument('--env_cfg', default='config/env.yaml', type=str)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint file')
    parser.add_argument('--n_episodes', default=10, type=int)
    args = parser.parse_args()

    with open(args.train_wm_cfg, 'r') as f:
        train_wm_cfg = yaml.safe_load(f)['train_wm']
    with open(args.train_agent_cfg, 'r') as f:
        train_agent_cfg = yaml.safe_load(f)['train_agent']
    with open(args.env_cfg, 'r') as f:
        env_cfg = yaml.safe_load(f)['env']

    # Extras

    DEVICE = 'cuda'

    # Enviroment parameters

    ENV_NAME = env_cfg['env_name']
    ENV_ACTIONS = env_n_actions(ENV_NAME)
    FRAMESKIP = env_cfg['frameskip']
    OBSERVATION_HEIGHT_WIDTH = env_cfg['observation_height_width']

    # World Model parameters (Categorical AutoEncoder + Transformer)

    TENSOR_DTYPE = torch.bfloat16 if train_wm_cfg['use_amp'] == True else torch.float16

    SEQUENCE_LENGTH = train_wm_cfg['sequence_length']

    LATENT_DIM = train_wm_cfg['latent_dim']
    CODES_PER_LATENT = train_wm_cfg['codes_per_latent']
    MODEL_DIM = train_wm_cfg['model_dim']

    N_TRANSFORMER_LAYERS = train_wm_cfg['n_transformer_layers']
    N_TRANSFORMER_HEADS = train_wm_cfg['n_transformer_heads']
    DROPOUT = train_wm_cfg['dropout']
    UP_PROJECTION_FACTOR = train_wm_cfg['up_projection_factor']

    # Actor Critic parameters

    CONTEXT_LENGTH = train_agent_cfg['enviroment_context_length']



    encoder = CategoricalEncoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    latent_action_embedder = LatentActionEmbedder(latent_dim=LATENT_DIM, 
                                                  codes_per_latent=CODES_PER_LATENT, 
                                                  env_actions=ENV_ACTIONS, 
                                                  embedding_dim=MODEL_DIM, 
                                                  sequence_length=SEQUENCE_LENGTH).to(DEVICE)
    transformer = TransformerDecoder(model_dim=MODEL_DIM, 
                                     n_transformer_layers=N_TRANSFORMER_LAYERS, 
                                     n_transformer_heads=N_TRANSFORMER_HEADS, 
                                     dropout=DROPOUT, 
                                     up_projection_factor=UP_PROJECTION_FACTOR, 
                                     latent_dim=LATENT_DIM, 
                                     codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    actor = Actor(latent_dim=LATENT_DIM, 
                  codes_per_latent=CODES_PER_LATENT,
                  embedding_dim=MODEL_DIM, 
                  env_actions=ENV_ACTIONS).to(DEVICE)

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)

    def clean_state_dict(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    encoder.load_state_dict(clean_state_dict(checkpoint['encoder']))
    latent_action_embedder.load_state_dict(clean_state_dict(checkpoint['latent_action_embedder']))
    transformer.load_state_dict(clean_state_dict(checkpoint['transformer']))
    actor.load_state_dict(clean_state_dict(checkpoint['actor']))

    encoder.eval()
    latent_action_embedder.eval()
    transformer.eval()
    actor.eval()

    rewards = []
    for ep in range(args.n_episodes):
        r = run_episode(env_name=ENV_NAME, 
                        frameskip=FRAMESKIP, 
                        observation_height_width=OBSERVATION_HEIGHT_WIDTH, 
                        actor=actor, 
                        encoder=encoder, 
                        latent_action_embedder=latent_action_embedder,
                        transformer=transformer, 
                        latent_dim=LATENT_DIM, 
                        codes_per_latent=CODES_PER_LATENT,
                        context_length=CONTEXT_LENGTH,
                        device=DEVICE, 
                        dtype=TENSOR_DTYPE)
        rewards.append(r)
        print(f"Episode {ep+1}: reward = {r}")

    print(f"\nMean reward over {args.n_episodes} episodes: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")