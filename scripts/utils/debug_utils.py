import torch
import os


def tensorboard_update(writer,
                       total_env_steps, 
                       world_model_loss, 
                       reconstruction_loss, 
                       rewards_loss, 
                       terminations_loss, 
                       dynamics_loss, 
                       dynamics_real_kl_div, 
                       representation_loss, 
                       representation_real_kl_div) -> None:
    writer.add_scalar('loss/total', world_model_loss.item(), total_env_steps)
    writer.add_scalar('loss/reconstruction', reconstruction_loss.item(), total_env_steps)
    writer.add_scalar('loss/reward', rewards_loss.item(), total_env_steps)
    writer.add_scalar('loss/termination', terminations_loss.item(), total_env_steps)
    writer.add_scalar('loss/dynamics', dynamics_loss.item(), total_env_steps)
    writer.add_scalar('loss/representation', representation_loss.item(), total_env_steps)
    writer.add_scalar('kl/dynamics', dynamics_real_kl_div.item(), total_env_steps)
    writer.add_scalar('kl/representation', representation_real_kl_div.item(), total_env_steps)


def save_checkpoint(encoder, decoder, storm_transformer, dist_head, 
                    reward_decoder, termination_decoder, 
                    actor, critic, ema_critic,
                    wm_optimizer, agent_optimizer, 
                    scaler, step, path="output/checkpoints"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'step': step,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'storm_transformer': storm_transformer.state_dict(),
        'dist_head': dist_head.state_dict(),
        'reward_decoder': reward_decoder.state_dict(),
        'termination_decoder': termination_decoder.state_dict(),
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'ema_critic': ema_critic.state_dict(),
        'wm_optimizer': wm_optimizer.state_dict(),
        'agent_optimizer': agent_optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }, os.path.join(path, f"checkpoint_step_{step}.pth"))