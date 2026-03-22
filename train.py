import torch
import argparse
import yaml
import os
import lpips
import copy
import time
import numpy as np
from scripts.utils.tensor_utils import EpochTimer, EMAScalar
from torch.utils.data import DataLoader, RandomSampler
from scripts.data_related.enviroment_steps import gather_steps
from scripts.data_related.atari_dataset import AtariDataset
from scripts.data_related.env_init import env_init
from scripts.utils.tensor_utils import env_n_actions
from scripts.utils.debug_utils import save_loss_history, plot_current_loss, save_checkpoint, generate_dataset_video
from scripts.models.categorical_vae.categorical_autoencoder_step import autoencoder_fwd_step
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.dynamics_modeling.dynamics_model_step import dm_fwd_step
from scripts.models.dynamics_modeling.total_loss import total_loss_step
from scripts.models.agent.train_agent import train_agent
from scripts.models.agent.critic import Critic
from scripts.models.agent.actor import Actor
from test import run_episode

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

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOT_TRAIN_STATUS = train_wm_cfg['plot_train_status']
    RUN_EVAL_EPISODES = train_wm_cfg['run_eval_episodes']
    N_EVAL_EPISODES = train_wm_cfg['n_eval_episodes']

    ENV_NAME = env_cfg['env_name']
    ENV_STEPS_PER_EPOCH = env_cfg['env_steps_per_epoch']
    ENV_ACTIONS = env_n_actions(ENV_NAME)

    FRAMESKIP = env_cfg['frameskip']
    NOOP_MAX = env_cfg['noop_max']
    EPISODIC_LIFE = env_cfg['episodic_life']
    MIN_REWARD = env_cfg['min_reward']
    MAX_REWARD = env_cfg['max_reward']
    OBSERVATION_HEIGHT_WIDTH = env_cfg['observation_height_width']

    EPOCHS = train_wm_cfg['epochs']
    TRAINING_STEPS_PER_EPOCH = train_wm_cfg['training_steps_per_epoch']
    WM_BATCH_SIZE = train_wm_cfg['wm_batch_size']
    AGENT_BATCH_SIZE = train_agent_cfg['agent_batch_size']
    SEQUENCE_LENGTH = train_wm_cfg['sequence_length']
    CONTEXT_LENGTH = train_agent_cfg['context_length']

    WORLD_MODEL_LEARNING_RATE = train_wm_cfg['world_model_learning_rate']
    DATASET_NUM_WORKERS = train_wm_cfg['dataloader_num_workers']

    LATENT_DIM = train_wm_cfg['latent_dim']
    CODES_PER_LATENT = train_wm_cfg['codes_per_latent']

    EMBEDDING_DIM = train_wm_cfg['embedding_dim']
    NUM_BLOCKS = train_wm_cfg['num_blocks']
    SLSTM_AT = train_wm_cfg['slstm_at']
    DROPOUT = train_wm_cfg['dropout']
    ADD_POST_BLOCKS_NORM = train_wm_cfg['add_post_blocks_norm']
    CONV1D_KERNEL_SIZE = train_wm_cfg['conv1d_kernel_size']
    QKV_PROJ_BLOCKSIZE = train_wm_cfg['qkv_proj_blocksize']
    NUM_HEADS = train_wm_cfg['num_heads']
    BIAS_INIT = train_wm_cfg['bias_init']
    PROJ_FACTOR = train_wm_cfg['proj_factor']
    ACT_FN = train_wm_cfg['act_fn']

    IMAGINATION_HORIZON = train_agent_cfg['imagination_horizon']
    GAMMA = train_agent_cfg['gamma']
    LAMBDA = train_agent_cfg['lambda']
    NABLA = train_agent_cfg['nabla']
    EMA_SIGMA = train_agent_cfg['ema_sigma']
    AGENT_LEARNING_RATE = train_agent_cfg['learning_rate']

    WM_DATALOADER_NUM_WORKERS = train_wm_cfg['dataloader_num_workers']
    AGENT_DATALOADER_NUM_WORKERS = train_agent_cfg['dataloader_num_workers']

    categorical_encoder = CategoricalEncoder(latent_dim=LATENT_DIM, 
                                             codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    categorical_decoder = CategoricalDecoder(latent_dim=LATENT_DIM, 
                                             codes_per_latent=CODES_PER_LATENT).to(DEVICE)
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
    lpips_model = lpips.LPIPS(net='alex').to(DEVICE).requires_grad_(False).eval()

    critic = Critic(latent_dim=LATENT_DIM, 
                    codes_per_latent=CODES_PER_LATENT, 
                    embedding_dim=EMBEDDING_DIM).to(DEVICE)
    
    ema_critic = copy.deepcopy(critic).requires_grad_(False).to(DEVICE)

    lowerbound_ema = EMAScalar(decay=0.99)
    upperbound_ema = EMAScalar(decay=0.99)

    actor = Actor(latent_dim=LATENT_DIM, 
                  codes_per_latent=CODES_PER_LATENT, 
                  embedding_dim=EMBEDDING_DIM, 
                  env_actions=ENV_ACTIONS).to(DEVICE)

    OPTIMIZER = torch.optim.Adam(list(categorical_encoder.parameters()) + 
                                 list(categorical_decoder.parameters()) +
                                 list(tokenizer.parameters()) + 
                                 list(xlstm_dm.parameters()),
                                 lr=WORLD_MODEL_LEARNING_RATE)
    
    AGENT_OPTIMIZER = torch.optim.Adam(list(critic.parameters()) +
                                       list(actor.parameters()),  
                                       lr=AGENT_LEARNING_RATE, 
                                       eps=1e-5)

    SCALER = torch.amp.GradScaler(enabled=True)

    categorical_encoder = torch.compile(categorical_encoder)
    categorical_decoder = torch.compile(categorical_decoder)
    # xlstm_dm = torch.compile(xlstm_dm) # Cannot compile because cluster gpu's do not support it.
    actor = torch.compile(actor)
    critic = torch.compile(critic)
    ema_critic = torch.compile(ema_critic)

    wm_dataset = AtariDataset(sequence_length=SEQUENCE_LENGTH)
    agent_dataset = AtariDataset(sequence_length=CONTEXT_LENGTH)
    
    env, last_observation, last_action, lives, features, state = env_init(env_name=ENV_NAME, 
                                                                         noop_max=NOOP_MAX, 
                                                                         frame_skip=FRAMESKIP, 
                                                                         screen_size=OBSERVATION_HEIGHT_WIDTH, 
                                                                         terminal_on_life_loss=EPISODIC_LIFE, 
                                                                         min_reward=MIN_REWARD, 
                                                                         max_reward=MAX_REWARD, 
                                                                         tokenizer=tokenizer, 
                                                                         encoder=categorical_encoder, 
                                                                         latent_dim=LATENT_DIM, 
                                                                         codes_per_latent=CODES_PER_LATENT, 
                                                                         device=DEVICE, 
                                                                         xlstm_dm=xlstm_dm)

    timers = EpochTimer()
    training_steps_finished = 0
    state = {}
    for epoch in range(EPOCHS):
        timers.reset()
        t0 = time.perf_counter()
        observations, actions, rewards, terminations, last_observation, last_action, lives, features, state = gather_steps(env=env, 
                                                                                                                                observation=last_observation, 
                                                                                                                                action=last_action, 
                                                                                                                                lives=lives,
                                                                                                                                features=features, 
                                                                                                                                state=state, 
                                                                                                                                env_steps_per_epoch=ENV_STEPS_PER_EPOCH, 
                                                                                                                                actor=actor, 
                                                                                                                                encoder=categorical_encoder, 
                                                                                                                                tokenizer=tokenizer, 
                                                                                                                                xlstm_dm=xlstm_dm, 
                                                                                                                                latent_dim=LATENT_DIM, 
                                                                                                                                codes_per_latent=CODES_PER_LATENT, 
                                                                                                                                device=DEVICE, 
                                                                                                                                context_length=CONTEXT_LENGTH, 
                                                                                                                                embedding_dim=EMBEDDING_DIM)

        wm_dataset.update(observations=observations, 
                          actions=actions, 
                          rewards=rewards, 
                          terminations=terminations)
        agent_dataset.update(observations=observations, 
                             actions=actions, 
                             rewards=rewards, 
                             terminations=terminations)
        wm_dataloader = DataLoader(dataset=wm_dataset, 
                                   batch_size=WM_BATCH_SIZE, 
                                   sampler=RandomSampler(data_source=wm_dataset, replacement=True, num_samples=WM_BATCH_SIZE*TRAINING_STEPS_PER_EPOCH), 
                                   num_workers=WM_DATALOADER_NUM_WORKERS, 
                                   pin_memory=True,
                                   persistent_workers=False, 
                                   drop_last=True)
        agent_dataloader = DataLoader(dataset=agent_dataset, 
                                      batch_size=AGENT_BATCH_SIZE, 
                                      sampler=RandomSampler(data_source=agent_dataset, replacement=True, num_samples=AGENT_BATCH_SIZE*TRAINING_STEPS_PER_EPOCH), 
                                      num_workers=WM_DATALOADER_NUM_WORKERS, 
                                      pin_memory=True,
                                      persistent_workers=False, 
                                      drop_last=True)
        wm_data_iterator = iter(wm_dataloader)
        agent_data_iterator = iter(agent_dataloader)
        timers.data_init = time.perf_counter() - t0

        epoch_loss_history = []
        for step in range(TRAINING_STEPS_PER_EPOCH):
            t0 = time.perf_counter()
            batch = next(wm_data_iterator)
            observations_batch, actions_batch, rewards_batch, terminations_batch = [x.to(DEVICE, non_blocking=True) for x in batch]
            timers.batch_extract += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            reconstruction_loss, latents_sampled_batch, posterior_logits = autoencoder_fwd_step(categorical_encoder=categorical_encoder, 
                                                                                                categorical_decoder=categorical_decoder, 
                                                                                                observations_batch=observations_batch, 
                                                                                                wm_batch_size=WM_BATCH_SIZE, 
                                                                                                sequence_length=SEQUENCE_LENGTH, 
                                                                                                latent_dim=LATENT_DIM, 
                                                                                                codes_per_latent=CODES_PER_LATENT,
                                                                                                lpips_loss_fn=lpips_model)
            timers.ae_fwd += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            tokens_batch = tokenizer.forward(latents_sampled_batch=latents_sampled_batch.detach(), actions_batch=actions_batch)
            timers.tokenizer += time.perf_counter() - t0

            t0 = time.perf_counter()
            rewards_loss, terminations_loss, dynamics_loss, dynamics_real_kl_div, representation_loss, representation_real_kl_div = dm_fwd_step(dynamics_model=xlstm_dm,
                                                                                                                                                latents_batch=latents_sampled_batch, 
                                                                                                                                                tokens_batch=tokens_batch, 
                                                                                                                                                rewards_batch=rewards_batch, 
                                                                                                                                                terminations_batch=terminations_batch, 
                                                                                                                                                batch_size=WM_BATCH_SIZE, 
                                                                                                                                                sequence_length=SEQUENCE_LENGTH, 
                                                                                                                                                latent_dim=LATENT_DIM, 
                                                                                                                                                codes_per_latent=CODES_PER_LATENT, 
                                                                                                                                                posterior_logits=posterior_logits)
            timers.dm_fwd += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            mean_total_loss = total_loss_step(reconstruction_loss=reconstruction_loss, 
                                              reward_loss=rewards_loss, 
                                              termination_loss=terminations_loss, 
                                              dynamics_loss=dynamics_loss, 
                                              representation_loss=representation_loss, 
                                              categorical_encoder=categorical_encoder, 
                                              categorical_decoder=categorical_decoder, 
                                              tokenizer=tokenizer, 
                                              dynamics_model=xlstm_dm, 
                                              optimizer=OPTIMIZER, 
                                              scaler=SCALER)
            timers.loss_calc += time.perf_counter() - t0

            t0 = time.perf_counter()
            batch = next(agent_data_iterator)
            observations_batch, actions_batch, rewards_batch, terminations_batch = [x.to(DEVICE, non_blocking=True) for x in batch]
            timers.agent_batch += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            mean_actor_loss, mean_critic_loss, mean_entropy, S_metric, norm_ratio_metric = train_agent(observations_batch=observations_batch, 
                                                                                                      actions_batch=actions_batch, 
                                                                                                      context_length=CONTEXT_LENGTH, 
                                                                                                      imagination_horizon=IMAGINATION_HORIZON, 
                                                                                                      env_actions=ENV_ACTIONS, 
                                                                                                      latent_dim=LATENT_DIM, 
                                                                                                      codes_per_latent=CODES_PER_LATENT,
                                                                                                      agent_batch_size=AGENT_BATCH_SIZE, 
                                                                                                      categorical_encoder=categorical_encoder,  
                                                                                                      tokenizer=tokenizer, 
                                                                                                      xlstm_dm=xlstm_dm, 
                                                                                                      actor=actor, 
                                                                                                      critic=critic,
                                                                                                      ema_critic=ema_critic,
                                                                                                      device=DEVICE, 
                                                                                                      gamma=GAMMA, 
                                                                                                      lambda_p=LAMBDA, 
                                                                                                      ema_sigma=EMA_SIGMA, 
                                                                                                      nabla=NABLA, 
                                                                                                      optimizer=AGENT_OPTIMIZER, 
                                                                                                      scaler=SCALER, 
                                                                                                      lowerbound_ema=lowerbound_ema, 
                                                                                                      upperbound_ema=upperbound_ema)
            timers.agent_train += time.perf_counter() - t0

            training_steps_finished += 1
                
            if training_steps_finished % 10**4 == 0:
                save_checkpoint(encoder=categorical_encoder,
                                decoder=categorical_decoder,
                                tokenizer=tokenizer,
                                dynamics=xlstm_dm,
                                actor=actor,
                                critic=critic,
                                ema_critic=ema_critic, 
                                wm_optimizer=OPTIMIZER, 
                                agent_optimizer=AGENT_OPTIMIZER, 
                                scaler=SCALER,
                                step=training_steps_finished, 
                                path=os.path.join(RUN_DIR, "checkpoints"))
                
            t0 = time.perf_counter()
            all_episodes_mean_reward = None
            if RUN_EVAL_EPISODES == True and training_steps_finished % 10**4 == 0:
                episode_mean_rewards = []
                for episode in range(N_EVAL_EPISODES):
                    _, _, all_rewards, _ = run_episode(env_name=ENV_NAME, 
                                                       frameskip=FRAMESKIP, 
                                                       noop_max=NOOP_MAX, 
                                                       episodic_life=EPISODIC_LIFE, 
                                                       min_reward=MIN_REWARD, 
                                                       max_reward=MAX_REWARD, 
                                                       observation_height_width=OBSERVATION_HEIGHT_WIDTH, 
                                                       actor=actor, 
                                                       encoder=categorical_encoder, 
                                                       tokenizer=tokenizer, 
                                                       xlstm_dm=xlstm_dm, 
                                                       latent_dim=LATENT_DIM, 
                                                       codes_per_latent=CODES_PER_LATENT, 
                                                       device=DEVICE, 
                                                       context_length=CONTEXT_LENGTH)
                    episode_mean_rewards.append(np.sum(all_rewards))
                
                all_episodes_mean_reward = np.mean(np.array(episode_mean_rewards))
            timers.eval_episodes += time.perf_counter() - t0
            
            step_metrics = {
                'reconstruction': reconstruction_loss.item(),
                'reward': rewards_loss.item(),
                'termination': terminations_loss.item(),
                'dynamics': dynamics_loss.item(),
                'dynamics_kl_div': dynamics_real_kl_div.item(), 
                'representation': representation_loss.item(), 
                'representation_kl_div': representation_real_kl_div.item(),
                'actor': mean_actor_loss,
                'critic': mean_critic_loss,
                'entropy': mean_entropy,
                'S': S_metric,
                'norm_ratio': norm_ratio_metric,
                'mean_episode_reward': all_episodes_mean_reward
            }
            
            
            epoch_loss_history.append(step_metrics)
        
        save_loss_history(new_losses=epoch_loss_history, output_dir=os.path.join(RUN_DIR, "logs"))
        if PLOT_TRAIN_STATUS == True:
            t0 = time.perf_counter()
            plot_current_loss(training_steps_per_epoch=TRAINING_STEPS_PER_EPOCH, epochs=EPOCHS, output_dir=os.path.join(RUN_DIR, "logs"))
            timers.plot = time.perf_counter() - t0

        timers.report(epoch)
