import torch
import torch.nn.functional as F
from torch.distributions import  OneHotCategorical
from typing import Tuple


def latent_unimix(posterior_raw_logits:torch.Tensor, uniform_mixture_percentage:float) -> torch.Tensor:
    number_of_latents = posterior_raw_logits.shape[-1]
    posterior_logits = F.softmax(posterior_raw_logits, dim=-1)
    uniform_mixture = uniform_mixture_percentage*torch.ones_like(posterior_logits)/number_of_latents
    nn_mixture = (1.0-uniform_mixture_percentage)*posterior_logits
    posterior_logits = torch.log(uniform_mixture+nn_mixture)
    return posterior_logits


def sample_with_straight_through_gradients(posterior_logits:torch.Tensor) -> torch.Tensor:
    one_hot_distribution = OneHotCategorical(logits=posterior_logits)
    return one_hot_distribution.sample() + one_hot_distribution.probs - one_hot_distribution.probs.detach()


def sample(posterior_raw_logits:torch.Tensor) -> Tuple:
    uniform_mixture_percentage = 0.01
    posterior_logits = latent_unimix(posterior_raw_logits=posterior_raw_logits, uniform_mixture_percentage=uniform_mixture_percentage)
    posterior_sample = sample_with_straight_through_gradients(posterior_logits=posterior_logits)
    return posterior_sample, posterior_logits