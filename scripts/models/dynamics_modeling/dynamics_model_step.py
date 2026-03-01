import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, cross_entropy
from typing import Tuple
from .xlstm_dm import XLSTM_DM


def dm_fwd_step(dynamics_model:XLSTM_DM, 
                latents_batch:torch.Tensor,
                tokens_batch:torch.Tensor, 
                rewards_batch:torch.Tensor, 
                terminations_batch:torch.Tensor, 
                batch_size:int, 
                sequence_length:int,
                latent_dim:int, 
                codes_per_latent:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    dynamics_model.train()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        next_latents_pred, rewards_pred, terminations_pred, features = dynamics_model.forward(tokens_batch=tokens_batch)

        next_latents_pred = next_latents_pred.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))
        latents_batch = latents_batch.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))

        time_shifted_preds = next_latents_pred[:, :-1].reshape(-1, codes_per_latent)
        time_shifted_targets = latents_batch[:, 1:].reshape(-1, codes_per_latent).detach()

        # rewards_loss = mse_loss(input=rewards_pred.squeeze(dim=-1), target=rewards_batch.float())
        # terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred.squeeze(dim=-1), target=terminations_batch.float())

        rewards_loss = mse_loss(input=rewards_pred[:, :-1].squeeze(dim=-1), target=rewards_batch[:, 1:].float())
        terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred[:, :-1].squeeze(dim=-1), target=terminations_batch[:, 1:].float())

        dynamics_loss = cross_entropy(input=time_shifted_preds, target=time_shifted_targets)
            
    return rewards_loss, terminations_loss, dynamics_loss