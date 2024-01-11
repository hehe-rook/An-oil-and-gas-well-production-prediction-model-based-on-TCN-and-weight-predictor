import torch.nn as nn
from DilatedConvBlock import DilatedConvBlock

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_sizes, dropout):
        super(TCN, self).__init__()
        num_levels = len(num_channels)
        layers = []
        for i in range(num_levels):
            dilation_rate = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            layers.append(DilatedConvBlock(in_channels, out_channels, dilation_rate, kernel_size, dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        out = self.tcn(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out
