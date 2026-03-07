import torch
import torch.nn as nn


def actor_loss(batch_lambda_returns:torch.Tensor, 
               state_values:torch.Tensor, 
               log_policy:torch.Tensor, 
               nabla:float, 
               entropy:torch.Tensor, 
               norm_ratio:torch.Tensor) -> float:
    
    advantage = (batch_lambda_returns - state_values).detach() / norm_ratio
    loss = -1*advantage*log_policy - nabla*entropy
    return loss.mean()


class Actor(nn.Module):
    def __init__(self, latent_dim, codes_per_latent, embedding_dim, env_actions) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.codes_per_latent = codes_per_latent
        self.embedding_dim = embedding_dim
        self.concat_dim = latent_dim*codes_per_latent+embedding_dim
        self.env_actions = env_actions

        self.linear_group_1 = nn.Sequential(nn.Linear(in_features=self.concat_dim, out_features=self.embedding_dim, bias=False), 
                                            nn.LayerNorm(normalized_shape=self.embedding_dim), 
                                            nn.ReLU())

        self.linear_group_2 = nn.Sequential(nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False), 
                                            nn.LayerNorm(normalized_shape=self.embedding_dim), 
                                            nn.ReLU())

        self.linear_3 = nn.Linear(in_features=self.embedding_dim, out_features=self.env_actions)

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        logits = self.linear_group_1(state)
        logits = self.linear_group_2(logits)
        logits = self.linear_3(logits)
        return logits
