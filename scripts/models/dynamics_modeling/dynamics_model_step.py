import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, cross_entropy
from typing import Tuple
from .xlstm_dm import XLSTM_DM
from torch.distributions import  OneHotCategorical
from einops import reduce
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.no_grad()
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymLogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        target = symlog(target)
        return 0.5*F.mse_loss(output, target)


class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes, lower_bound, upper_bound):
        super().__init__()
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes-1)

        # use register buffer so that bins move with .cuda() automatically
        self.bins: torch.Tensor
        self.register_buffer(
            'bins', torch.linspace(-20, 20, num_classes), persistent=False)

    def forward(self, output, target):
        target = symlog(target)
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = torch.bucketize(target, self.bins)
        diff = target - self.bins[index-1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = torch.clamp(weight, 0, 1)
        weight = weight.unsqueeze(-1)

        target_prob = (1-weight)*F.one_hot(index-1, self.num_classes) + weight*F.one_hot(index, self.num_classes)

        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        return loss.mean()

    def decode(self, output):
        return symexp(F.softmax(output, dim=-1) @ self.bins)


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
    symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20).to(device='cuda')
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        next_latents_pred, rewards_pred, terminations_pred, features = dynamics_model.forward(tokens_batch=tokens_batch)

        next_latents_pred = next_latents_pred.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))
        latents_batch = latents_batch.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))
        posterior_logits = posterior_logits.view(batch_size, sequence_length, latent_dim, codes_per_latent)

        # time_shifted_preds = next_latents_pred[:, :-1].reshape(-1, codes_per_latent)
        # time_shifted_targets = latents_batch[:, 1:].reshape(-1, codes_per_latent).detach()

        # prior_logits = next_latents_pred
        # post_logits = posterior_logits

        # rewards_loss = symlog_twohot_loss_func(rewards_pred.squeeze(dim=-1).to(device='cuda'), rewards_batch.float().to(device='cuda'))
        # terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred.squeeze(dim=-1), target=terminations_batch.float())
        
        rewards_loss = symlog_twohot_loss_func(rewards_pred[:, :-1].squeeze(dim=-1).to(device='cuda'), rewards_batch[:, 1:].float().to(device='cuda'))
        terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred[:, :-1].squeeze(dim=-1), target=terminations_batch[:, 1:].float())


        # dynamics_loss, dynamics_real_kl_div = categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
        # representation_loss, representation_real_kl_div = categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())

        # dynamics_loss = cross_entropy(input=time_shifted_preds, target=time_shifted_targets)
        prior_logits = next_latents_pred[:, :-1]
        post_logits = posterior_logits[:, 1:].detach()

        prior_flat = prior_logits.reshape(-1, codes_per_latent)
        post_probs_flat = F.softmax(post_logits, dim=-1).reshape(-1, codes_per_latent)

        ce = cross_entropy(input=prior_flat, target=post_probs_flat, reduction='none')
        dynamics_loss = ce.view(batch_size, sequence_length - 1, latent_dim).sum(dim=-1).mean()

            
    # return rewards_loss, terminations_loss, dynamics_loss, dynamics_real_kl_div, representation_loss, representation_real_kl_div
    return rewards_loss, terminations_loss, dynamics_loss