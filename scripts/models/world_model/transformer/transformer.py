import torch
import torch.nn as nn
from typing import Tuple


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim:int, n_transformer_heads:int, dropout:float, up_projection_factor:int) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.n_transformer_heads = n_transformer_heads

        self.dropout = nn.Dropout(dropout)

        self.d_k = model_dim//n_transformer_heads
        self.d_v = self.d_k
        
        self.W_Q = nn.Linear(in_features=self.model_dim, out_features=self.d_k*n_transformer_heads)
        self.W_K = nn.Linear(in_features=self.model_dim, out_features=self.d_k*n_transformer_heads)
        self.W_V = nn.Linear(in_features=self.model_dim, out_features=self.d_v*n_transformer_heads)

        self.softmax = nn.Softmax(dim=-1)

        self.W_O = nn.Linear(in_features=n_transformer_heads*self.d_v, out_features=model_dim)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=model_dim)

        self.linear_1 = nn.Linear(in_features=model_dim, out_features=model_dim*up_projection_factor)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=model_dim*up_projection_factor, out_features=model_dim)

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=model_dim)


    def forward(self, query:torch.Tensor, 
                      key:torch.Tensor, 
                      value:torch.Tensor, 
                      batch_size:int, 
                      sequence_length:int, 
                      mask:torch.Tensor) -> Tuple:
        
        Q = self.W_Q(query).view(batch_size, sequence_length, self.n_transformer_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, sequence_length, self.n_transformer_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, sequence_length, self.n_transformer_heads, self.d_v).transpose(1, 2)

        masked_scaled_matmul = (torch.matmul(Q, K.transpose(2, 3))/self.d_k**0.5).masked_fill(mask == 0, -1e9)
        attention = torch.matmul(self.dropout(self.softmax(masked_scaled_matmul)), V)

        concat_heads = attention.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.n_transformer_heads*self.d_v)
        multi_head_attention = self.dropout(self.W_O(concat_heads))

        residual_norm_1 = self.layer_norm_1(multi_head_attention + query)
        # feedforward = self.linear_2(self.dropout(self.relu(self.linear_1(residual_norm_1))))
        feedforward = self.linear_2(self.relu(self.linear_1(residual_norm_1)))
        residual_norm_2 = self.layer_norm_2(self.dropout(feedforward) + residual_norm_1)

        return residual_norm_2, K, V
    

    def forward_kv_cache(self, query:torch.Tensor, 
                               key:torch.Tensor, 
                               value:torch.Tensor,
                               kv_cache:tuple, 
                               batch_size:int, 
                               sequence_length:int) -> Tuple:
        
        Q = self.W_Q(query).view(batch_size, sequence_length, self.n_transformer_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, sequence_length, self.n_transformer_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, sequence_length, self.n_transformer_heads, self.d_v).transpose(1, 2)

        K = torch.concat(tensors=(kv_cache[0], K), dim=2)
        V = torch.concat(tensors=(kv_cache[1], V), dim=2)

        scaled_matmul = torch.matmul(Q, K.transpose(2, 3))/self.d_k**0.5
        attention = torch.matmul(self.dropout(self.softmax(scaled_matmul)), V)

        concat_heads = attention.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.n_transformer_heads*self.d_v)
        multi_head_attention = self.dropout(self.W_O(concat_heads))

        residual_norm_1 = self.layer_norm_1(multi_head_attention + query)
        # feedforward = self.linear_2(self.dropout(self.relu(self.linear_1(residual_norm_1))))
        feedforward = self.linear_2(self.relu(self.linear_1(residual_norm_1)))
        residual_norm_2 = self.layer_norm_2(self.dropout(feedforward) + residual_norm_1)

        return residual_norm_2, K, V


class TransformerDecoder(nn.Module):
    def __init__(self, model_dim:int, 
                       n_transformer_layers:int, 
                       n_transformer_heads:int, 
                       dropout:float, 
                       up_projection_factor:int, 
                       latent_dim:int, 
                       codes_per_latent:int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                model_dim=model_dim, 
                n_transformer_heads=n_transformer_heads, 
                dropout=dropout, 
                up_projection_factor=up_projection_factor
            ) for layer in range(n_transformer_layers)
        ])

        self.prior_raw_logits = nn.Linear(in_features=model_dim, out_features=latent_dim*codes_per_latent)

        self.reward_head = nn.Sequential(nn.Linear(in_features=model_dim, out_features=model_dim, bias=False), 
                                         nn.LayerNorm(model_dim), 
                                         nn.ReLU(inplace=True), 
                                         nn.Linear(in_features=model_dim, out_features=model_dim, bias=False), 
                                         nn.LayerNorm(model_dim), 
                                         nn.ReLU(inplace=True), 
                                         nn.Linear(in_features=model_dim, out_features=255))

        self.termination_head = nn.Sequential(nn.Linear(in_features=model_dim, out_features=model_dim, bias=False), 
                                              nn.LayerNorm(model_dim), 
                                              nn.ReLU(inplace=True), 
                                              nn.Linear(in_features=model_dim, out_features=model_dim, bias=False), 
                                              nn.LayerNorm(model_dim), 
                                              nn.ReLU(inplace=True), 
                                              nn.Linear(in_features=model_dim, out_features=1))
        
        self.kv_cache = None

    
    def reset_kv_cache(self) -> None:
        self.kv_cache = None


    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> Tuple:

        batch_size = x.shape[0]
        sequence_length = x.shape[1]
    
        KV_n_layers = []
        for layer in self.layers:
            x, K, V = layer(
                            query=x, 
                            key=x, 
                            value=x, 
                            batch_size=batch_size, 
                            sequence_length=sequence_length, 
                            mask=mask
                        )
            KV_n_layers.append((K, V))

        prior_raw_logits = self.prior_raw_logits(x)
        reward_logits = self.reward_head(x)
        termination_logits = self.termination_head(x)
       
        return prior_raw_logits, reward_logits, termination_logits, x, KV_n_layers
    

    def forward_kv_cache(self, x:torch.Tensor, mask:torch.Tensor) -> Tuple:
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        if self.kv_cache is None:
            prior_raw_logits, reward_logits, termination_logits, x, KV_n_layers = self.forward(x=x, mask=mask)
            self.kv_cache = KV_n_layers
        else:
            for layer_idx, layer in enumerate(self.layers):
                x, K, V = layer.forward_kv_cache(query=x, 
                                                 key=x, 
                                                 value=x, 
                                                 kv_cache=self.kv_cache[layer_idx],
                                                 batch_size=batch_size, 
                                                 sequence_length=sequence_length)
                self.kv_cache[layer_idx] = (K, V)

            prior_raw_logits = self.prior_raw_logits(x)
            reward_logits = self.reward_head(x)
            termination_logits = self.termination_head(x)

        return prior_raw_logits, reward_logits, termination_logits, x