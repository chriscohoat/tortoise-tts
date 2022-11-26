import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import os


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )
    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        out = fused_leaky_relu(out, self.bias * self.lr_mul)
        return out


class RandomLatentConverter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(*[EqualLinear(channels, channels, lr_mul=.1) for _ in range(5)],
                                    nn.Linear(channels, channels))
        self.channels = channels

    def forward(self, ref):
        r = torch.randn(ref.shape[0], self.channels, device=ref.device)
        y = self.layers(r)
        return y


class RandomButDeterministicLatentConverter(nn.Module):
    def __init__(self, seed, channels):
        super().__init__()
        self.seed = seed
        torch.manual_seed(seed)
        self.layers = nn.Sequential(*[EqualLinear(channels, channels, lr_mul=.1) for _ in range(5)],
                                    nn.Linear(channels, channels))
        self.channels = channels

    def forward(self, ref):
        torch.manual_seed(self.seed)
        r = torch.randn(ref.shape[0], self.channels, device=ref.device)
        # Hacky, but to test the seeding and determinism, we output this random tensor
        # to a file named time.time() + '.txt'
        # This is to make sure that the random tensor is the same for each run
        # and that the seed is working
        # Go up one directory from this file
        current_file = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_file)
        # Create a file with the name of the current time
        output_txt_path = os.path.join(parent_dir, str(time.time()) + '.txt')
        with open(output_txt_path, 'w') as f:
            f.write(str(r))
        print(f'Wrote random tensor to {output_txt_path}')
        y = self.layers(r)
        return y + ref


if __name__ == '__main__':
    model = RandomLatentConverter(512)
    model(torch.randn(5,512))
