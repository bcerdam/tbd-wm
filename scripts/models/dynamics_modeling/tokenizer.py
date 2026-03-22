import torch
import torch.nn as nn


class Tokenizer(nn.Module):
    def __init__(self, latent_dim:int, codes_per_latent:int, env_actions:int, embedding_dim:int, sequence_length:int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.codes_per_latent = codes_per_latent
        self.env_actions = env_actions
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        
        self.linear_1 = nn.Linear(in_features=self.latent_dim*self.codes_per_latent+self.env_actions, 
                                  out_features=self.embedding_dim)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.ReLU = nn.ReLU()

        self.linear_2 = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.embedding_dim)

    def forward(self, latents_sampled_batch:torch.Tensor, actions_batch:torch.Tensor) -> torch.Tensor:
        latents_actions_tensor = torch.cat(tensors=(latents_sampled_batch.flatten(start_dim=2), actions_batch), dim=2)
        linear_1 = self.linear_1(latents_actions_tensor)
        layer_norm_1 = self.layer_norm_1(linear_1)
        relu = self.ReLU(layer_norm_1)

        linear_2 = self.linear_2(relu)
        layer_norm_2 = self.layer_norm_2(linear_2)

        return layer_norm_2