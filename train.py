import torch
import argparse
import yaml
import os
import copy
import ale_py
import shutil
import numpy as np
import gymnasium as gym
from scripts.utils.tensor_utils import EMAScalar, reshape_observation
from scripts.utils.tensor_utils import env_n_actions, MaxLast2FrameSkipWrapper, LifeLossInfo
from scripts.utils.debug_utils import tensorboard_update, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.world_model.categorical_autoencoder.encoder import CategoricalEncoder
from scripts.models.world_model.categorical_autoencoder.decoder import CategoricalDecoder
from scripts.models.world_model.transformer.latent_action_embedder import LatentActionEmbedder
from scripts.models.world_model.transformer.transformer import TransformerDecoder
from scripts.models.world_model.world_model_training_step import world_model_training_step
from scripts.models.agent.train_agent import train_agent, take_action
from scripts.models.agent.critic import Critic
from scripts.models.agent.actor import Actor
from collections import deque
from evaluation import run_episode
import warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_wm_cfg', default='config/train_wm.yaml', type=str, help='Path to train wm parameters .yaml file')
    parser.add_argument('--train_agent_cfg', default='config/train_agent.yaml', type=str, help='Path to train agent parameters .yaml file')
    parser.add_argument('--env_cfg', default='config/env.yaml', type=str, help='Path to env parameters .yaml file')
    parser.add_argument('--run_dir', default='output/run', type=str)
    args, unknown = parser.parse_known_args()

    RUN_DIR = args.run_dir
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR)
    writer = SummaryWriter(log_dir=os.path.join(RUN_DIR, "tensorboard"))

    with open(args.train_wm_cfg, 'r') as file_train_wm, open(args.env_cfg, 'r') as file_env, open(args.train_agent_cfg, 'r') as file_train_agent:
        train_wm_cfg = yaml.safe_load(file_train_wm)['train_wm']
        train_agent_cfg = yaml.safe_load(file_train_agent)['train_agent']
        env_cfg = yaml.safe_load(file_env)['env']

    configs = {'train_wm': train_wm_cfg, 'train_agent': train_agent_cfg, 'env': env_cfg}
    i = 0
    while i < len(unknown):
        key = unknown[i].lstrip('--')
        val = yaml.safe_load(unknown[i+1])
        cfg_name, param_name = key.split('.')
        configs[cfg_name][param_name] = val
        i += 2

    # Extras

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUN_EVAL_EPISODES = train_wm_cfg['run_eval_episodes']
    N_EVAL_EPISODES = train_wm_cfg['n_eval_episodes']

    # Enviroment parameters

    ENV_NAME = env_cfg['env_name']
    TOTAL_ENV_STEPS = env_cfg['total_env_steps']
    ENV_ACTIONS = env_n_actions(ENV_NAME)
    FRAMESKIP = env_cfg['frameskip']
    OBSERVATION_HEIGHT_WIDTH = env_cfg['observation_height_width']

    # World Model parameters (Categorical AutoEncoder + Transformer)

    TENSOR_DTYPE = torch.bfloat16 if train_wm_cfg['use_amp'] == True else torch.float16

    WM_BATCH_SIZE = train_wm_cfg['wm_batch_size']
    SEQUENCE_LENGTH = train_wm_cfg['sequence_length']
    WORLD_MODEL_LEARNING_RATE = train_wm_cfg['world_model_learning_rate']

    LATENT_DIM = train_wm_cfg['latent_dim']
    CODES_PER_LATENT = train_wm_cfg['codes_per_latent']

    MODEL_DIM = train_wm_cfg['model_dim']
    N_TRANSFORMER_LAYERS = train_wm_cfg['n_transformer_layers']
    N_TRANSFORMER_HEADS = train_wm_cfg['n_transformer_heads']
    DROPOUT = train_wm_cfg['dropout']
    UP_PROJECTION_FACTOR = train_wm_cfg['up_projection_factor']

    # Actor Critic parameters

    AGENT_BATCH_SIZE = train_agent_cfg['agent_batch_size']
    ENVIROMENT_CONTEXT_LENGTH = train_agent_cfg['enviroment_context_length']
    IMAGINATION_CONTEXT_LENGTH = train_agent_cfg['imagination_context_length']
    AGENT_LEARNING_RATE = train_agent_cfg['learning_rate']

    IMAGINATION_HORIZON = train_agent_cfg['imagination_horizon']
    GAMMA = train_agent_cfg['gamma']
    LAMBDA = train_agent_cfg['lambda']
    NABLA = train_agent_cfg['nabla']
    EMA_SIGMA = train_agent_cfg['ema_sigma']

    categorical_encoder = CategoricalEncoder(latent_dim=LATENT_DIM, 
                                             codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    categorical_decoder = CategoricalDecoder(latent_dim=LATENT_DIM, 
                                             codes_per_latent=CODES_PER_LATENT).to(DEVICE)
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
    critic = Critic(latent_dim=LATENT_DIM, 
                    codes_per_latent=CODES_PER_LATENT, 
                    embedding_dim=MODEL_DIM).to(DEVICE)
    ema_critic = copy.deepcopy(critic).requires_grad_(False).to(DEVICE)
    actor = Actor(latent_dim=LATENT_DIM, 
                codes_per_latent=CODES_PER_LATENT, 
                embedding_dim=MODEL_DIM, 
                env_actions=ENV_ACTIONS).to(DEVICE)

    lowerbound_ema = EMAScalar(decay=0.99)
    upperbound_ema = EMAScalar(decay=0.99)

    WORLD_MODEL_OPTIMIZER = torch.optim.Adam(list(categorical_encoder.parameters()) + 
                                             list(categorical_decoder.parameters()) + 
                                             list(latent_action_embedder.parameters()) + 
                                             list(transformer.parameters()),
                                             lr=WORLD_MODEL_LEARNING_RATE)
    
    AGENT_OPTIMIZER = torch.optim.Adam(list(critic.parameters()) +
                                       list(actor.parameters()),  
                                       lr=AGENT_LEARNING_RATE, 
                                       eps=1e-5)

    WM_SCALER = torch.amp.GradScaler(enabled=True)
    AGENT_SCALER = torch.amp.GradScaler(enabled=True)

    categorical_encoder = torch.compile(categorical_encoder)
    categorical_decoder = torch.compile(categorical_decoder)
    latent_action_embedder = torch.compile(latent_action_embedder)
    transformer = torch.compile(transformer)
    actor = torch.compile(actor)
    critic = torch.compile(critic)
    ema_critic = torch.compile(ema_critic)

    wm_dataset = AtariDataset(sequence_length=SEQUENCE_LENGTH, total_env_steps=TOTAL_ENV_STEPS, env_actions=ENV_ACTIONS, device=DEVICE, dtype=TENSOR_DTYPE)
    agent_dataset = AtariDataset(sequence_length=IMAGINATION_CONTEXT_LENGTH, total_env_steps=TOTAL_ENV_STEPS, env_actions=ENV_ACTIONS, device=DEVICE, dtype=TENSOR_DTYPE)

    gym.register_envs(ale_py)
    env = gym.make(id=ENV_NAME, frameskip=1, full_action_space=False, render_mode="rgb_array")
    env = MaxLast2FrameSkipWrapper(env, skip=FRAMESKIP)
    env = gym.wrappers.ResizeObservation(env, shape=(OBSERVATION_HEIGHT_WIDTH, OBSERVATION_HEIGHT_WIDTH))
    env = LifeLossInfo(env)

    observation, info = env.reset()
    observation = reshape_observation(observation=observation)
    action = env.action_space.sample()

    context_obs = deque(maxlen=ENVIROMENT_CONTEXT_LENGTH)
    context_act = deque(maxlen=ENVIROMENT_CONTEXT_LENGTH)  
    for env_step in range(TOTAL_ENV_STEPS):
        context_obs.append(observation)
        context_act.append(action)

        next_observation, reward, termination, truncated, info = env.step(action)
        termination = np.logical_or(termination, info["life_loss"])

        wm_dataset.update(observation=observation, 
                          action=action, 
                          reward=reward, 
                          termination=termination)
        agent_dataset.update(observation=observation, 
                             action=action, 
                             reward=reward, 
                             termination=termination)
        
        action = take_action(context_obs=context_obs, 
                             context_act=context_act, 
                             categorical_encoder=categorical_encoder, 
                             latent_action_embedder=latent_action_embedder, 
                             transformer=transformer, 
                             actor=actor, 
                             latent_dim=LATENT_DIM, 
                             codes_per_latent=CODES_PER_LATENT, 
                             env_actions=ENV_ACTIONS, 
                             device=DEVICE, 
                             tensor_dtype=TENSOR_DTYPE)

        observation = next_observation
        observation = reshape_observation(observation)
        
        if env_step >= AGENT_BATCH_SIZE:

            categorical_encoder.train()
            categorical_decoder.train()
            latent_action_embedder.train()
            transformer.train()
            observations_batch, actions_batch, rewards_batch, terminations_batch = wm_dataset.extract_random_batch(batch_size=WM_BATCH_SIZE)
            world_model_loss, reconstruction_loss, rewards_loss, terminations_loss, dynamics_loss, dynamics_real_kl_div, representation_loss, representation_real_kl_div = world_model_training_step(observations_batch=observations_batch, 
                                                                                                                                                                                                     actions_batch=actions_batch, 
                                                                                                                                                                                                     rewards_batch=rewards_batch, 
                                                                                                                                                                                                     terminations_batch=terminations_batch, 
                                                                                                                                                                                                     categorical_encoder=categorical_encoder, 
                                                                                                                                                                                                     categorical_decoder=categorical_decoder, 
                                                                                                                                                                                                     latent_action_embedder=latent_action_embedder, 
                                                                                                                                                                                                     transformer=transformer, 
                                                                                                                                                                                                     wm_batch_size=WM_BATCH_SIZE, 
                                                                                                                                                                                                     sequence_length=SEQUENCE_LENGTH, 
                                                                                                                                                                                                     latent_dim=LATENT_DIM, 
                                                                                                                                                                                                     codes_per_latent=CODES_PER_LATENT, 
                                                                                                                                                                                                     optimizer=WORLD_MODEL_OPTIMIZER, 
                                                                                                                                                                                                     scaler=WM_SCALER)

            categorical_encoder.eval()
            categorical_decoder.eval()
            latent_action_embedder.eval()
            transformer.eval()
            save_video = (env_step % 2000 == 0 and env_step > 0)
            observations_batch, actions_batch, rewards_batch, terminations_batch = agent_dataset.extract_random_batch(batch_size=AGENT_BATCH_SIZE)
            mean_actor_loss, mean_critic_loss, mean_entropy, S_metric, norm_ratio_metric = train_agent(observations_batch=observations_batch, 
                                                                                                       actions_batch=actions_batch, 
                                                                                                       context_length=IMAGINATION_CONTEXT_LENGTH, 
                                                                                                       imagination_horizon=IMAGINATION_HORIZON, 
                                                                                                       latent_dim=LATENT_DIM, 
                                                                                                       codes_per_latent=CODES_PER_LATENT, 
                                                                                                       agent_batch_size=AGENT_BATCH_SIZE, 
                                                                                                       categorical_encoder=categorical_encoder, 
                                                                                                       categorical_decoder=categorical_decoder, 
                                                                                                       transformer=transformer, 
                                                                                                       latent_action_embedder=latent_action_embedder, 
                                                                                                       actor=actor, 
                                                                                                       critic=critic, 
                                                                                                       ema_critic=ema_critic, 
                                                                                                       device=DEVICE, 
                                                                                                       gamma=GAMMA, 
                                                                                                       lambda_p=LAMBDA, 
                                                                                                       ema_sigma=EMA_SIGMA, 
                                                                                                       nabla=NABLA, 
                                                                                                       optimizer=AGENT_OPTIMIZER, 
                                                                                                       scaler=AGENT_SCALER, 
                                                                                                       lowerbound_ema=lowerbound_ema, 
                                                                                                       upperbound_ema=upperbound_ema, 
                                                                                                       save_video=save_video, 
                                                                                                       run_dir=RUN_DIR, 
                                                                                                       env_step=env_step)
            
            all_episodes_mean_reward = None
            if RUN_EVAL_EPISODES == True and env_step % 2500 == 0:
                actor.eval()
                episode_mean_rewards = []
                for episode in range(N_EVAL_EPISODES):
                    total_reward = run_episode(env_name=ENV_NAME, 
                                               frameskip=FRAMESKIP, 
                                               observation_height_width=OBSERVATION_HEIGHT_WIDTH,
                                               actor=actor, 
                                               categorical_encoder=categorical_encoder, 
                                               latent_action_embedder=latent_action_embedder, 
                                               transformer=transformer, 
                                               latent_dim=LATENT_DIM, 
                                               codes_per_latent=CODES_PER_LATENT, 
                                               context_length=ENVIROMENT_CONTEXT_LENGTH, 
                                               device=DEVICE, 
                                               dtype=TENSOR_DTYPE)
                    episode_mean_rewards.append(total_reward)
                all_episodes_mean_reward = np.mean(np.array(episode_mean_rewards))

            # tensorboard --logdir output/run/tensorboard
            tensorboard_update(writer=writer, 
                               total_env_steps=env_step, 
                               world_model_loss=world_model_loss, 
                               reconstruction_loss=reconstruction_loss, 
                               rewards_loss=rewards_loss, 
                               terminations_loss=terminations_loss, 
                               dynamics_loss=dynamics_loss, 
                               dynamics_real_kl_div=dynamics_real_kl_div, 
                               representation_loss=representation_loss, 
                               representation_real_kl_div=representation_real_kl_div, 
                               actor_loss=mean_actor_loss, 
                               critic_loss=mean_critic_loss, 
                               entropy=mean_entropy, 
                               S=S_metric, 
                               norm_ratio=norm_ratio_metric, 
                               episode_mean_rewards=all_episodes_mean_reward)
                    
        if termination == True or truncated == True:
            observation, info = env.reset()
            observation = reshape_observation(observation=observation)
            action = env.action_space.sample()
            context_obs.clear()
            context_act.clear()

        if env_step % 10**4 == 0:
            save_checkpoint(encoder=categorical_encoder,
                            decoder=categorical_decoder,
                            transformer=transformer,
                            actor=actor,
                            critic=critic,
                            ema_critic=ema_critic, 
                            wm_optimizer=WORLD_MODEL_OPTIMIZER, 
                            agent_optimizer=AGENT_OPTIMIZER, 
                            wm_scaler=WM_SCALER,
                            agent_scaler=AGENT_SCALER, 
                            step=env_step, 
                            path=os.path.join(RUN_DIR, "checkpoints"))
            