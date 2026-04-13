import torch
import argparse
import yaml
import os
import copy
import time
import numpy as np
import gymnasium as gym
import ale_py
from scripts.utils.tensor_utils import EMAScalar, normalize_observation, reshape_observation
from torch.utils.data import DataLoader, RandomSampler
# from scripts.data_related.enviroment_steps import gather_steps
from scripts.data_related.atari_dataset import AtariDataset
from scripts.utils.tensor_utils import env_n_actions
from scripts.utils.debug_utils import save_loss_history, plot_current_loss
from scripts.models.world_model.world_model_training_step import world_model_training_step
# from scripts.models.dynamics_modeling.dynamics_model_step import dm_fwd_step
# from scripts.models.dynamics_modeling.total_loss import total_loss_step
# from scripts.models.agent.train_agent import train_agent
from scripts.models.agent.critic import Critic
from scripts.models.agent.actor import Actor
# from test import run_episode
from scripts.models.world_model.categorical_autoencoder.encoder import CategoricalEncoder
from scripts.models.world_model.categorical_autoencoder.decoder import CategoricalDecoder
from scripts.models.world_model.transformer.latent_action_embedder import LatentActionEmbedder
from scripts.models.world_model.transformer.transformer import TransformerDecoder


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
    os.makedirs(RUN_DIR, exist_ok=True)

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

    TENSOR_DTYPE = torch.bfloat16 if train_wm_cfg['use_amp'] == True else torch.float32

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

    # Rewrite once transformer is functional
    # OPTIMIZER = torch.optim.Adam(list(categorical_encoder.parameters()) + 
    #                              list(categorical_decoder.parameters()) +
    #                              list(storm_transformer.parameters()) + 
    #                              list(dist_head.parameters()) + 
    #                              list(reward_decoder.parameters()) + 
    #                              list(termination_decoder.parameters()),
    #                              lr=WORLD_MODEL_LEARNING_RATE)

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
    env = gym.make(id=ENV_NAME, frameskip=FRAMESKIP, full_action_space=False, render_mode="rgb_array")
    env = gym.wrappers.ResizeObservation(env, shape=(OBSERVATION_HEIGHT_WIDTH, OBSERVATION_HEIGHT_WIDTH))
    observation, info = env.reset() # o_0
    observation = reshape_observation(normalize_observation(observation))

    for env_step in range(TOTAL_ENV_STEPS):

        action = env.action_space.sample() # a_0
        action_array = np.zeros(ENV_ACTIONS)
        action_array[action] = 1.0

        next_observation, reward, termination, truncated, info = env.step(action) # o_1, r_0, t_0

        wm_dataset.update(observation=observation, 
                          action=action, 
                          reward=reward, 
                          termination=termination)
        agent_dataset.update(observation=observation, 
                             action=action, 
                             reward=reward, 
                             termination=termination)
        
        if env_step >= AGENT_BATCH_SIZE:

            observations_batch, actions_batch, rewards_batch, terminations_batch = wm_dataset.extract_random_batch(batch_size=WM_BATCH_SIZE)

            # Train World Model (Create single script  for this)
            world_model_loss = world_model_training_step(observations_batch=observations_batch, 
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
                                                         scaler=AGENT_SCALER)
            


            # Train Agent (Create single script  for this)
                # State: (z_prior_t+1, h_t) -> a_t+1

            if env_step % 500 == 0:
                print(world_model_loss)
        
        observation = next_observation
        observation = reshape_observation(normalize_observation(observation))

        if termination == True or truncated == True:
            observation, info = env.reset()
            observation = reshape_observation(normalize_observation(observation))


    # for epoch in range(EPOCHS):
    #     timers.reset()
    #     t0 = time.perf_counter()
    #     categorical_encoder.eval()
    #     storm_transformer.eval()
    #     dist_head.eval()
    #     actor.eval()
    #     observations, actions, rewards, terminations, last_observation, last_action, lives, features, state = gather_steps(env=env, 
    #                                                                                                                             observation=last_observation, 
    #                                                                                                                             action=last_action, 
    #                                                                                                                             lives=lives,
    #                                                                                                                             features=features, 
    #                                                                                                                             state=state, 
    #                                                                                                                             env_steps_per_epoch=ENV_STEPS_PER_EPOCH, 
    #                                                                                                                             actor=actor, 
    #                                                                                                                             encoder=categorical_encoder, 
    #                                                                                                                             latent_dim=LATENT_DIM, 
    #                                                                                                                             codes_per_latent=CODES_PER_LATENT, 
    #                                                                                                                             device=DEVICE, 
    #                                                                                                                             context_length=CONTEXT_LENGTH, 
    #                                                                                                                             embedding_dim=EMBEDDING_DIM, 
    #                                                                                                                             storm_transformer=storm_transformer)

    #     wm_dataset.update(observations=observations, 
    #                       actions=actions, 
    #                       rewards=rewards, 
    #                       terminations=terminations)
    #     agent_dataset.update(observations=observations, 
    #                          actions=actions, 
    #                          rewards=rewards, 
    #                          terminations=terminations)
    #     wm_dataloader = DataLoader(dataset=wm_dataset, 
    #                                batch_size=WM_BATCH_SIZE, 
    #                                sampler=RandomSampler(data_source=wm_dataset, replacement=True, num_samples=WM_BATCH_SIZE*TRAINING_STEPS_PER_EPOCH), 
    #                                num_workers=WM_DATALOADER_NUM_WORKERS, 
    #                                pin_memory=True,
    #                                persistent_workers=False, 
    #                                drop_last=True)
    #     agent_dataloader = DataLoader(dataset=agent_dataset, 
    #                                   batch_size=AGENT_BATCH_SIZE, 
    #                                   sampler=RandomSampler(data_source=agent_dataset, replacement=True, num_samples=AGENT_BATCH_SIZE*TRAINING_STEPS_PER_EPOCH), 
    #                                   num_workers=WM_DATALOADER_NUM_WORKERS, 
    #                                   pin_memory=True,
    #                                   persistent_workers=False, 
    #                                   drop_last=True)
    #     wm_data_iterator = iter(wm_dataloader)
    #     agent_data_iterator = iter(agent_dataloader)
    #     timers.data_init = time.perf_counter() - t0

    #     epoch_loss_history = []
    #     for step in range(TRAINING_STEPS_PER_EPOCH):
    #         t0 = time.perf_counter()
    #         batch = next(wm_data_iterator)
    #         observations_batch, actions_batch, rewards_batch, terminations_batch = [x.to(DEVICE, non_blocking=True) for x in batch]
    #         timers.batch_extract += time.perf_counter() - t0

    #         categorical_encoder.train()
    #         categorical_decoder.train()
    #         storm_transformer.train()
    #         dist_head.train()
    #         reward_decoder.train()
    #         termination_decoder.train()
            
    #         t0 = time.perf_counter()
    #         reconstruction_loss, latents_sampled_batch, posterior_logits = autoencoder_fwd_step(categorical_encoder=categorical_encoder, 
    #                                                                                             categorical_decoder=categorical_decoder, 
    #                                                                                             observations_batch=observations_batch, 
    #                                                                                             wm_batch_size=WM_BATCH_SIZE, 
    #                                                                                             sequence_length=SEQUENCE_LENGTH, 
    #                                                                                             latent_dim=LATENT_DIM, 
    #                                                                                             codes_per_latent=CODES_PER_LATENT)
    #         timers.ae_fwd += time.perf_counter() - t0
            
    #         t0 = time.perf_counter()
    #         # tokens_batch = tokenizer.forward(latents_sampled_batch=latents_sampled_batch.detach(), actions_batch=actions_batch)
    #         timers.tokenizer += time.perf_counter() - t0

    #         t0 = time.perf_counter()
    #         rewards_loss, terminations_loss, dynamics_loss, dynamics_real_kl_div, representation_loss, representation_real_kl_div = dm_fwd_step(dynamics_model=storm_transformer,
    #                                                                                                                                             latents_batch=latents_sampled_batch, 
    #                                                                                                                                             actions_batch=actions_batch, 
    #                                                                                                                                             rewards_batch=rewards_batch, 
    #                                                                                                                                             terminations_batch=terminations_batch, 
    #                                                                                                                                             batch_size=WM_BATCH_SIZE, 
    #                                                                                                                                             sequence_length=SEQUENCE_LENGTH, 
    #                                                                                                                                             latent_dim=LATENT_DIM, 
    #                                                                                                                                             codes_per_latent=CODES_PER_LATENT, 
    #                                                                                                                                             posterior_logits=posterior_logits,
    #                                                                                                                                             dist_head=dist_head, 
    #                                                                                                                                             reward_decoder=reward_decoder, 
    #                                                                                                                                             termination_decoder=termination_decoder)
    #         timers.dm_fwd += time.perf_counter() - t0
            
    #         t0 = time.perf_counter()
    #         mean_total_loss = total_loss_step(reconstruction_loss=reconstruction_loss, 
    #                                           reward_loss=rewards_loss, 
    #                                           termination_loss=terminations_loss, 
    #                                           dynamics_loss=dynamics_loss, 
    #                                           representation_loss=representation_loss, 
    #                                           categorical_encoder=categorical_encoder, 
    #                                           categorical_decoder=categorical_decoder, 
    #                                           dynamics_model=storm_transformer,
    #                                           dist_head=dist_head, 
    #                                           reward_decoder=reward_decoder, 
    #                                           termination_decoder=termination_decoder,  
    #                                           optimizer=OPTIMIZER, 
    #                                           scaler=WM_SCALER)
    #         timers.loss_calc += time.perf_counter() - t0

    #         t0 = time.perf_counter()
    #         batch = next(agent_data_iterator)
    #         observations_batch, actions_batch, rewards_batch, terminations_batch = [x.to(DEVICE, non_blocking=True) for x in batch]
    #         timers.agent_batch += time.perf_counter() - t0
            
    #         t0 = time.perf_counter()
    #         mean_actor_loss, mean_critic_loss, mean_entropy, S_metric, norm_ratio_metric = train_agent(observations_batch=observations_batch, 
    #                                                                                                   actions_batch=actions_batch, 
    #                                                                                                   context_length=CONTEXT_LENGTH, 
    #                                                                                                   imagination_horizon=IMAGINATION_HORIZON, 
    #                                                                                                   env_actions=ENV_ACTIONS, 
    #                                                                                                   latent_dim=LATENT_DIM, 
    #                                                                                                   codes_per_latent=CODES_PER_LATENT,
    #                                                                                                   agent_batch_size=AGENT_BATCH_SIZE, 
    #                                                                                                   categorical_encoder=categorical_encoder,  
    #                                                                                                   storm_transformer=storm_transformer, 
    #                                                                                                   dist_head=dist_head, 
    #                                                                                                   reward_decoder=reward_decoder, 
    #                                                                                                   termination_decoder=termination_decoder,
    #                                                                                                   actor=actor, 
    #                                                                                                   critic=critic,
    #                                                                                                   ema_critic=ema_critic,
    #                                                                                                   device=DEVICE, 
    #                                                                                                   gamma=GAMMA, 
    #                                                                                                   lambda_p=LAMBDA, 
    #                                                                                                   ema_sigma=EMA_SIGMA, 
    #                                                                                                   nabla=NABLA, 
    #                                                                                                   optimizer=AGENT_OPTIMIZER, 
    #                                                                                                   scaler=AGENT_SCALER, 
    #                                                                                                   lowerbound_ema=lowerbound_ema, 
    #                                                                                                   upperbound_ema=upperbound_ema)
    #         timers.agent_train += time.perf_counter() - t0

    #         training_steps_finished += 1
                
    #         # if training_steps_finished % 10**4 == 0:
    #         #     save_checkpoint(encoder=categorical_encoder,
    #         #                     decoder=categorical_decoder,
    #         #                     storm_transformer=storm_transformer,
    #         #                     dist_head=dist_head,
    #         #                     reward_decoder=reward_decoder,
    #         #                     termination_decoder=termination_decoder,
    #         #                     actor=actor,
    #         #                     critic=critic,
    #         #                     ema_critic=ema_critic, 
    #         #                     wm_optimizer=OPTIMIZER, 
    #         #                     agent_optimizer=AGENT_OPTIMIZER, 
    #         #                     scaler=SCALER,
    #         #                     step=training_steps_finished, 
    #         #                     path=os.path.join(RUN_DIR, "checkpoints"))
                
    #         t0 = time.perf_counter()
    #         t0 = time.perf_counter()
    #         all_episodes_mean_reward = None
    #         if RUN_EVAL_EPISODES == True and training_steps_finished % 2500 == 0:
    #             categorical_encoder.eval()
    #             storm_transformer.eval()
    #             actor.eval()
            
    #             episode_mean_rewards = []
    #             for episode in range(N_EVAL_EPISODES):
    #                 _, _, all_rewards, _ = run_episode(env_name=ENV_NAME, 
    #                                                    frameskip=FRAMESKIP, 
    #                                                    noop_max=NOOP_MAX, 
    #                                                    episodic_life=EPISODIC_LIFE, 
    #                                                    min_reward=MIN_REWARD, 
    #                                                    max_reward=MAX_REWARD, 
    #                                                    observation_height_width=OBSERVATION_HEIGHT_WIDTH, 
    #                                                    actor=actor, 
    #                                                    encoder=categorical_encoder, 
    #                                                    storm_transformer=storm_transformer, 
    #                                                    latent_dim=LATENT_DIM, 
    #                                                    codes_per_latent=CODES_PER_LATENT, 
    #                                                    device=DEVICE, 
    #                                                    context_length=CONTEXT_LENGTH)
    #                 episode_mean_rewards.append(np.sum(all_rewards))
                
    #             all_episodes_mean_reward = np.mean(np.array(episode_mean_rewards))
    #         timers.eval_episodes += time.perf_counter() - t0
            
    #         step_metrics = {
    #             'reconstruction': reconstruction_loss.item(),
    #             'reward': rewards_loss.item(),
    #             'termination': terminations_loss.item(),
    #             'dynamics': dynamics_loss.item(),
    #             'dynamics_kl_div': dynamics_real_kl_div.item(), 
    #             'representation': representation_loss.item(), 
    #             'representation_kl_div': representation_real_kl_div.item(),
    #             'actor': mean_actor_loss,
    #             'critic': mean_critic_loss,
    #             'entropy': mean_entropy,
    #             'S': S_metric,
    #             'norm_ratio': norm_ratio_metric,
    #             'mean_episode_reward': all_episodes_mean_reward
    #         }
            
            
    #         epoch_loss_history.append(step_metrics)
        
    #     save_loss_history(new_losses=epoch_loss_history, output_dir=os.path.join(RUN_DIR, "logs"))
    #     if PLOT_TRAIN_STATUS == True:
    #         t0 = time.perf_counter()
    #         plot_current_loss(training_steps_per_epoch=TRAINING_STEPS_PER_EPOCH, epochs=EPOCHS, output_dir=os.path.join(RUN_DIR, "logs"))
    #         timers.plot = time.perf_counter() - t0

    #     timers.report(epoch)