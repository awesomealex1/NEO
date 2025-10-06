import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class NEO(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype)
        
        self.sample_count = 0
        self.global_mean = torch.zeros(self.in_features, device=device)
    
    def forward(self, input: Tensor) -> Tensor:
        # If model device is changed, global_mean device is not changed.
        # Registering global_mean as a parameter would fix, but model loading is more complicated.
        if input.device != self.global_mean.device:
            self.global_mean = self.global_mean.to(input.device)
        
        batch_sample_count = input.size(0)
        self.sample_count += batch_sample_count
        seen_weight = (self.sample_count - batch_sample_count) / self.sample_count
        unseen_weight = batch_sample_count / self.sample_count
        self.global_mean = seen_weight * self.global_mean + unseen_weight * torch.mean(input, dim=0)
        
        return F.linear(input - self.global_mean, self.weight, self.bias)


class NEO_Continual(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        alpha: int = 0.1
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype)
        
        self.alpha = alpha
        self.sample_count = 0
        self.global_mean = torch.zeros(self.in_features, device=device)
    
    def forward(self, input: Tensor) -> Tensor:
        # If model device is changed, global_mean device is not changed.
        # Registering global_mean as a parameter would fix, but model loading is more complicated.
        if input.device != self.global_mean.device:
            self.global_mean = self.global_mean.to(input.device)
        self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * torch.mean(input, dim=0)
        return F.linear(input - self.global_mean, self.weight, self.bias)