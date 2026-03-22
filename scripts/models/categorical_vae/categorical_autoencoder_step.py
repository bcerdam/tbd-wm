import torch
import torch.nn as nn
import lpips
from .encoder import CategoricalEncoder
from .decoder import CategoricalDecoder
from .sampler import sample
from typing import Tuple
import torch.nn.functional as F
from einops import reduce



class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


def autoencoder_fwd_step(categorical_encoder:CategoricalEncoder, 
                         categorical_decoder:CategoricalDecoder, 
                         observations_batch:torch.Tensor, 
                         wm_batch_size:int, 
                         sequence_length:int, 
                         latent_dim:int, 
                         codes_per_latent:int) -> Tuple:
    
    mse_loss_func = MSELoss()
    
    categorical_encoder.train()
    categorical_decoder.train()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latents_batch = categorical_encoder.forward(observations_batch=observations_batch, 
                                                    batch_size=wm_batch_size, 
                                                    sequence_length=sequence_length, 
                                                    latent_dim=latent_dim, 
                                                    codes_per_latent=codes_per_latent)    

        latents_sampled_batch = sample(latents_batch=latents_batch, batch_size=wm_batch_size, sequence_length=sequence_length)

        reconstructed_observations_batch = categorical_decoder.forward(latents_batch=latents_sampled_batch, 
                                                                        batch_size=wm_batch_size, 
                                                                        sequence_length=sequence_length, 
                                                                        latent_dim=latent_dim, 
                                                                        codes_per_latent=codes_per_latent)
        
        reconstruction_loss = mse_loss_func.forward(obs_hat=reconstructed_observations_batch, obs=observations_batch)
    
    return reconstruction_loss, latents_sampled_batch, latents_batch