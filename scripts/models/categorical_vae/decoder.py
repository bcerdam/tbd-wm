import torch
import torch.nn as nn
from typing import Tuple


def upscale_layer(in_channels:int, 
                  out_channels:int, 
                  kernel_size: int, 
                  stride: int, 
                  padding: int) -> Tuple:
    
    conv = nn.ConvTranspose2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              bias=False)
    
    bn = nn.BatchNorm2d(num_features=out_channels)
    ReLU = nn.ReLU(inplace=True)
    return conv, bn, ReLU


class CategoricalDecoder(nn.Module):
    def __init__(self, latent_dim:int, codes_per_latent:int) -> None:
        super().__init__()
        self.channels = [256, 128, 64, 32, 3]
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.linear_in_dim = latent_dim*codes_per_latent

        self.flattened_latent = nn.Flatten(start_dim=1, end_dim=-1)

        self.current_dim = 64
        layers = []
        for channel in range(len(self.channels)-1):
            self.current_dim = self.current_dim // (self.kernel_size-self.stride)
            conv, bn, ReLU = upscale_layer(in_channels=self.channels[channel],
                                           out_channels=self.channels[channel+1], 
                                           kernel_size=self.kernel_size, 
                                           stride=self.stride,
                                           padding=self.padding)
            
            if self.channels[channel] != 32:
                layers.extend([conv, bn, ReLU])
            else:
                conv = nn.ConvTranspose2d(in_channels=self.channels[channel], 
                                            out_channels=self.channels[channel+1], 
                                            kernel_size=self.kernel_size, 
                                            stride=self.stride, 
                                            padding=self.padding, 
                                            bias=True)
                layers.extend([conv])

        self.upscale_features = nn.Sequential(*layers)
        linear_out_dim = self.channels[0]*self.current_dim*self.current_dim
        self.linear = nn.Linear(in_features=self.linear_in_dim, out_features=linear_out_dim, bias=False) # bias = True
        self.projected_bn = nn.BatchNorm2d(num_features=self.channels[0])
        self.projected_relu = nn.ReLU(inplace=True)


    def forward(self, latents_batch: torch.Tensor, 
                      batch_size:int, 
                      sequence_length:int, 
                      latent_dim:int, 
                      codes_per_latent:int) -> torch.Tensor:
        
        latents_batch = latents_batch.view(batch_size*sequence_length, latent_dim, codes_per_latent)
        flattened_latents = self.flattened_latent(latents_batch)
        projected_features = self.linear(flattened_latents)
        reshaped_features = projected_features.reshape(-1, self.channels[0], self.current_dim, self.current_dim)
        reshaped_features = self.projected_relu(self.projected_bn(reshaped_features))
        upscaled_features = self.upscale_features(reshaped_features)
        upscaled_features = upscaled_features.view(batch_size, sequence_length, 3, 64, 64)
        return upscaled_features