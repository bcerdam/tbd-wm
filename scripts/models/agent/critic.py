import torch
import torch.nn as nn


def critic_loss(batch_lambda_returns:torch.Tensor, 
                state_values:torch.Tensor, 
                ema_lambda_returns:torch.Tensor, 
                symlog_twohot_loss) -> float:
    
    value_loss = symlog_twohot_loss(state_values, batch_lambda_returns.detach())
    slow_value_regularization_loss = symlog_twohot_loss(state_values, ema_lambda_returns.detach())

    return value_loss + slow_value_regularization_loss


class Critic(nn.Module):
    def __init__(self, latent_dim, codes_per_latent, embedding_dim) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        self.codes_per_latent = codes_per_latent
        self.embedding_dim = embedding_dim
        self.concat_dim = latent_dim*codes_per_latent+embedding_dim

        self.linear_group_1 = nn.Sequential(nn.Linear(in_features=self.concat_dim, out_features=self.embedding_dim, bias=False), 
                                           nn.LayerNorm(normalized_shape=self.embedding_dim), 
                                           nn.ReLU())

        self.linear_group_2 = nn.Sequential(nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False), 
                                            nn.LayerNorm(normalized_shape=self.embedding_dim), 
                                            nn.ReLU())

        self.linear_3 = nn.Linear(in_features=self.embedding_dim, out_features=255)

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        state_value = self.linear_group_1(state)
        state_value = self.linear_group_2(state_value)
        state_value = self.linear_3(state_value)
        return state_value