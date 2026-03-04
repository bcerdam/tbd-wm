import torch
import torch.nn.functional as F
import lpips
from .encoder import CategoricalEncoder
from .decoder import CategoricalDecoder
from .sampler import sample

import torch.nn as nn
from einops import rearrange, repeat, reduce

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
                         codes_per_latent:int) -> tuple[torch.Tensor, torch.Tensor]:
                         # lpips_loss_fn:lpips.LPIPS) -> tuple[torch.Tensor, torch.Tensor]:
    
    categorical_encoder.train()
    categorical_decoder.train()
    mse_loss = MSELoss()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latents_batch = categorical_encoder.forward(observations_batch=observations_batch, 
                                                    batch_size=wm_batch_size, 
                                                    sequence_length=sequence_length, 
                                                    latent_dim=latent_dim, 
                                                    codes_per_latent=codes_per_latent)    
        # latents_batch = (16, 64, 32, 32)    

        latents_sampled_batch = sample(latents_batch=latents_batch, batch_size=wm_batch_size, sequence_length=sequence_length)
        # (16, 64, 32, 32)

        reconstructed_observations_batch = categorical_decoder.forward(latents_batch=latents_sampled_batch, 
                                                                        batch_size=wm_batch_size, 
                                                                        sequence_length=sequence_length, 
                                                                        latent_dim=latent_dim, 
                                                                        codes_per_latent=codes_per_latent)
        # (16, 64, 3, 64, 64)
        
        # reconstruction_loss = F.mse_loss(reconstructed_observations_batch, wm_observations_batch)
        reconstruction_loss = ((reconstructed_observations_batch - observations_batch) ** 2).sum(dim=(-3, -2, -1)).mean()
        reconstruction_loss = mse_loss.forward(obs_hat=reconstructed_observations_batch, obs=observations_batch)
        # perceptual_loss = lpips_loss_fn(wm_observations_batch.view(-1, 3, 64, 64), reconstructed_observations_batch.view(-1, 3, 64, 64)).mean()
        # reconstruction_loss = reconstruction_loss + 0.2 * perceptual_loss
    
    return reconstruction_loss, latents_sampled_batch