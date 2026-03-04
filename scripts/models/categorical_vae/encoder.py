import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def downscale_layer(in_channels:int, 
                    out_channels:int, 
                    kernel_size: int, 
                    stride: int, 
                    padding: int) -> Tuple:
    
    conv = nn.Conv2d(in_channels=in_channels, 
                     out_channels=out_channels, 
                     kernel_size=kernel_size, 
                     stride=stride, 
                     padding=padding, 
                     bias=False)
    bn = nn.BatchNorm2d(num_features=out_channels)
    ReLU = nn.ReLU(inplace=True)
    return conv, bn, ReLU


class CategoricalEncoder(nn.Module):
    def __init__(self, latent_dim:int, codes_per_latent:int) -> None:
        super().__init__()
        self.input_shape = (3, 64, 64)
        self.output_shape = (latent_dim, codes_per_latent)
        self.channels = [3, 32, 64, 128, 256]
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.linear_projection_amount = latent_dim*codes_per_latent

        layers = []
        for channel in range(len(self.channels)-1):
            conv, bn, ReLU = downscale_layer(in_channels=self.channels[channel],
                                             out_channels=self.channels[channel+1], 
                                             kernel_size=self.kernel_size, 
                                             stride=self.stride,
                                             padding=self.padding)
            # layers.append(conv)
            # layers.append(bn)
            # layers.append(ReLU)
            layers.extend([conv, bn, ReLU])

        self.downscale_features = nn.Sequential(*layers)
        self.flattened_downscale_features = nn.Flatten(start_dim=1, end_dim=-1)

        with torch.no_grad():
            input_dummy = torch.zeros(1, *self.input_shape)
            output_dummy = self.downscale_features(input_dummy)
            flattened_output_dummy = self.flattened_downscale_features(output_dummy)
            flattened_in_features = flattened_output_dummy.shape[1]

        self.linear = nn.Linear(in_features=flattened_in_features, out_features=self.linear_projection_amount)


    def forward(self, observations_batch: torch.Tensor, 
                      batch_size:int, 
                      sequence_length:int, 
                      latent_dim:int, 
                      codes_per_latent:int) -> torch.Tensor:
        
        # (16*64, 3, 64, 64)
        observations_batch = observations_batch.view(batch_size*sequence_length, 3, 64, 64)        
        downscaled_features = self.downscale_features(observations_batch) # (16*64, 256, 4, 4)
        flattened_features = self.flattened_downscale_features(downscaled_features) # (16*64, 4096)
        projected_features = self.linear(flattened_features) # (16*64, 1024)
        reshaped_features = projected_features.view(batch_size, sequence_length, latent_dim, codes_per_latent)
        # reshaped_features = projected_features.reshape(-1, *self.output_shape) # (-1, 32, 32)
        # reshaped_features = reshaped_features.view(batch_size, sequence_length, latent_dim, codes_per_latent)
        return reshaped_features # (16, 64, 32, 32)