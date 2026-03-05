import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, cross_entropy
from typing import Tuple
from .xlstm_dm import XLSTM_DM
from torch.distributions import  OneHotCategorical
from einops import reduce
import torch.nn as nn


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.free_bits = 1

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


def dm_fwd_step(dynamics_model:XLSTM_DM, 
                latents_batch:torch.Tensor,
                tokens_batch:torch.Tensor, 
                rewards_batch:torch.Tensor, 
                terminations_batch:torch.Tensor, 
                batch_size:int, 
                sequence_length:int,
                latent_dim:int, 
                codes_per_latent:int, 
                posterior_logits:torch.Tensor) -> Tuple:

    dynamics_model.train()
    categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        next_latents_pred, rewards_pred, terminations_pred, features = dynamics_model.forward(tokens_batch=tokens_batch)

        next_latents_pred = next_latents_pred.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))
        latents_batch = latents_batch.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))

        # time_shifted_preds = next_latents_pred[:, :-1].reshape(-1, codes_per_latent)
        # time_shifted_targets = latents_batch[:, 1:].reshape(-1, codes_per_latent).detach()
        prior_logits = next_latents_pred
        post_logits = posterior_logits

        rewards_loss = mse_loss(input=rewards_pred[:, :-1].squeeze(dim=-1), target=rewards_batch[:, 1:].float())
        terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred[:, :-1].squeeze(dim=-1), target=terminations_batch[:, 1:].float())
        # dynamics_loss = cross_entropy(input=time_shifted_preds, target=time_shifted_targets)

        dynamics_loss, dynamics_real_kl_div = categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
        representation_loss, representation_real_kl_div = categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            
    return rewards_loss, terminations_loss, dynamics_loss, dynamics_real_kl_div, representation_loss, representation_real_kl_div