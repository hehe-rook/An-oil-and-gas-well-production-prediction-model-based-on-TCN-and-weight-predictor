import torch.nn as nn

class DilatedConvBlock(nn.Module):
    def __init__(self, input_size, output_size, dilation_rate, kernel_size, dropout):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, padding=(kernel_size - 1) // 2 * dilation_rate, dilation=dilation_rate)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out
