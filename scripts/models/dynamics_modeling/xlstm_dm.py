import torch
import torch.nn as nn
from typing import List, Tuple
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig, 
    sLSTMBlockConfig, 
    sLSTMLayerConfig, 
    FeedForwardConfig)


def xlstm_block_stack(sequence_length:int, 
                       num_blocks:int, 
                       embedding_dim:int, 
                       slstm_at:List, 
                       dropout:float, 
                       add_post_blocks_norm:bool, 
                       conv1d_kernel_size:int, 
                       qkv_proj_blocksize:int, 
                       num_heads:int, 
                       bias_init:str, 
                       proj_factor:float, 
                       act_fn:str) -> xLSTMBlockStack:
                      
    cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, 
                    qkv_proj_blocksize=qkv_proj_blocksize, 
                    num_heads=num_heads)),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla", 
                    num_heads=num_heads, 
                    conv1d_kernel_size=conv1d_kernel_size, 
                    bias_init=bias_init
                ),
                feedforward=FeedForwardConfig(
                    embedding_dim=embedding_dim, 
                    proj_factor=proj_factor, 
                    act_fn=act_fn
                )
            ),
            context_length=sequence_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=slstm_at,
            dropout=dropout,
            add_post_blocks_norm=add_post_blocks_norm)
    
    xlstm_stack = xLSTMBlockStack(cfg)
    return xlstm_stack


class XLSTM_DM(nn.Module):
    def __init__(self, 
                 sequence_length:int, 
                 num_blocks:int, 
                 embedding_dim:int, 
                 slstm_at:List, 
                 dropout:float, 
                 add_post_blocks_norm:bool, 
                 conv1d_kernel_size:int, 
                 qkv_proj_blocksize:int, 
                 num_heads:int, 
                 latent_dim:int, 
                 codes_per_latent:int, 
                 bias_init:str, 
                 proj_factor:float, 
                 act_fn:str) -> None:
        super().__init__()

        self.xlstm_stack = xlstm_block_stack(sequence_length=sequence_length, 
                                             num_blocks=num_blocks, 
                                             embedding_dim=embedding_dim, 
                                             slstm_at=slstm_at, 
                                             dropout=dropout, 
                                             add_post_blocks_norm=add_post_blocks_norm, 
                                             conv1d_kernel_size=conv1d_kernel_size, 
                                             qkv_proj_blocksize=qkv_proj_blocksize, 
                                             num_heads=num_heads, 
                                             bias_init=bias_init, 
                                             proj_factor=proj_factor, 
                                             act_fn=act_fn)
                
        self.latent_projection = nn.Linear(in_features=embedding_dim, out_features=latent_dim*codes_per_latent)

        self.reward_head_linear_1 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.reward_head_linear_2 = nn.Linear(in_features=embedding_dim, out_features=1)

        self.termination_head_1 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.termination_head_2 = nn.Linear(in_features=embedding_dim, out_features=1)


    def forward(self, tokens_batch:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.xlstm_stack(tokens_batch)

        next_state_latent = self.latent_projection(features)

        reward = self.reward_head_linear_1(features)
        reward = self.reward_head_linear_2(reward)

        termination = self.termination_head_1(features)
        termination = self.termination_head_2(termination)

        return next_state_latent, reward, termination, features
    
    
    def step(self, tokens_batch:torch.Tensor, state:dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        features, new_state = self.xlstm_stack.step(tokens_batch, state=state)

        next_state_latent = self.latent_projection(features)

        reward = self.reward_head_linear_1(features)
        reward = self.reward_head_linear_2(reward)

        termination = self.termination_head_1(features)
        termination = self.termination_head_2(termination)

        return next_state_latent, reward, termination, features, new_state