import torch
import torch.nn as nn


class LatentActionEmbedder(nn.Module):
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

        self.positional_encoding = nn.Embedding(num_embeddings=sequence_length, embedding_dim=embedding_dim)

    def forward(self, posterior_sample_batch:torch.Tensor, actions_batch:torch.Tensor, start_pos:int=0) -> torch.Tensor:
        latents_actions_tensor = torch.cat(tensors=(posterior_sample_batch.flatten(start_dim=2), actions_batch), dim=2)
        linear_1 = self.linear_1(latents_actions_tensor)
        layer_norm_1 = self.layer_norm_1(linear_1)
        relu = self.ReLU(layer_norm_1)

        linear_2 = self.linear_2(relu)
        layer_norm_2 = self.layer_norm_2(linear_2)

        seq_len = layer_norm_2.shape[1]
        positions = torch.arange(start_pos, start_pos + seq_len, device=layer_norm_2.device)
        positional_encoding = self.positional_encoding(positions)
        positional_encoding_batch = positional_encoding.unsqueeze(0).expand(posterior_sample_batch.shape[0], -1, -1)

        latent_action_embedding = layer_norm_2 + positional_encoding_batch

        return latent_action_embedding