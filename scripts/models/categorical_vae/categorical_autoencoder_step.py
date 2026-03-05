import torch
import lpips
from .encoder import CategoricalEncoder
from .decoder import CategoricalDecoder
from .sampler import sample
from typing import Tuple


def autoencoder_fwd_step(categorical_encoder:CategoricalEncoder, 
                         categorical_decoder:CategoricalDecoder, 
                         observations_batch:torch.Tensor, 
                         wm_batch_size:int, 
                         sequence_length:int, 
                         latent_dim:int, 
                         codes_per_latent:int,
                         lpips_loss_fn:lpips.LPIPS) -> Tuple:
    
    categorical_encoder.train()
    categorical_decoder.train()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
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
        
        reconstruction_loss = ((reconstructed_observations_batch - observations_batch) ** 2).sum(dim=(-3, -2, -1)).mean()
        perceptual_loss = lpips_loss_fn(observations_batch.view(-1, 3, 64, 64), reconstructed_observations_batch.view(-1, 3, 64, 64)).mean()
        reconstruction_loss = reconstruction_loss + 0.2 * perceptual_loss
    
    return reconstruction_loss, latents_sampled_batch, latents_batch