import torch
from .categorical_autoencoder.encoder import CategoricalEncoder
from .categorical_autoencoder.decoder import CategoricalDecoder
from .categorical_autoencoder.categorical_autoencoder_step import autoencoder_fwd_step


def world_model_training_step(observations_batch:torch.Tensor, 
                              actions_batch:torch.Tensor, 
                              rewards_batch:torch.Tensor, 
                              terminations_batch:torch.Tensor, 
                              categorical_encoder:CategoricalEncoder, 
                              categorical_decoder:CategoricalDecoder, 
                              wm_batch_size:int, 
                              sequence_length:int, 
                              latent_dim:int, 
                              codes_per_latent:int, 
                              optimizer:torch.optim.Optimizer, 
                              scaler:torch.amp.grad_scaler) -> None:
    

    reconstruction_loss, posterior_sample, posterior_logits = autoencoder_fwd_step(categorical_encoder=categorical_encoder, 
                                                                                   categorical_decoder=categorical_decoder, 
                                                                                   observations_batch=observations_batch, 
                                                                                   wm_batch_size=wm_batch_size, 
                                                                                   sequence_length=sequence_length, 
                                                                                   latent_dim=latent_dim, 
                                                                                   codes_per_latent=codes_per_latent)

    # Train Transformer (Create single script  for this)
        # Token (name?) -> (z_posterior_t, a_t) -> h_t -> z_prior_t+1

    # sum_of_losses = (reconstruction_loss+reward_loss+termination_loss+0.5*dynamics_loss+0.1*representation_loss)
    world_model_loss = (reconstruction_loss)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(world_model_loss).backward()
    scaler.unscale_(optimizer)
    
    # all_wm_params = list(categorical_encoder.parameters()) + \
    #             list(categorical_decoder.parameters()) + \
    #             list(dynamics_model.parameters()) + \
    #             list(dist_head.parameters()) + \
    #             list(reward_decoder.parameters()) + \
    #             list(termination_decoder.parameters())
    all_wm_params = list(categorical_encoder.parameters()) + \
                    list(categorical_decoder.parameters())
    torch.nn.utils.clip_grad_norm_(all_wm_params, 1000.0)
    
    scaler.step(optimizer)
    scaler.update()

    return world_model_loss