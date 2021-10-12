import torch
import torch.nn as nn

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_length, ratio=16):
        super(ChannelSpatialAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, max_length, ratio)
        self.spatial_attention = SpatialAttention(out_channels, max_length, kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, max_length, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(max_length) # [batch, in_channels, src_len]
        self.max_pool = nn.AdaptiveMaxPool1d(max_length) # [batch, in_channels, src_len]
        self.fc = nn.Sequential(nn.Conv1d(in_channels, in_channels//ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_channels//ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out) # [batch, in_channels, src_len]
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, max_length, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(max_length)
        self.max_pool = nn.AdaptiveMaxPool1d(max_length)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1) # 进行连接
        out = self.conv1(out) # [batch, 2 * in_channels, src_len]
        out = self.sigmoid(out)
        return out
