import torch
import torch.nn.functional as F
from torch.distributions import  OneHotCategorical


def latent_unimix(latents_batch:torch.Tensor, uniform_mixture_percentage:float) -> torch.Tensor:
    number_of_latents = latents_batch.shape[-1]
    stochastic_latents_batch = F.softmax(latents_batch, dim=-1)
    uniform_mixture = uniform_mixture_percentage*torch.ones_like(stochastic_latents_batch)/number_of_latents
    nn_mixture = (1.0-uniform_mixture_percentage)*stochastic_latents_batch
    log_probabilities = torch.log(uniform_mixture+nn_mixture)
    return log_probabilities


def sample_with_straight_through_gradients(log_probabilities:torch.Tensor) -> torch.Tensor:
    one_hot_distribution = OneHotCategorical(logits=log_probabilities)
    return one_hot_distribution.sample() + one_hot_distribution.probs - one_hot_distribution.probs.detach()


def sample(latents_batch:torch.Tensor, batch_size:int, sequence_length:int) -> torch.Tensor:
    uniform_mixture_percentage = 0.01
    log_probabilities = latent_unimix(latents_batch=latents_batch, uniform_mixture_percentage=uniform_mixture_percentage)
    return sample_with_straight_through_gradients(log_probabilities=log_probabilities)