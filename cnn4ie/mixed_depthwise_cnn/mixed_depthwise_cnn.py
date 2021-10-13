import torch
import torch.nn as nn

import numpy as np

class GroupMixedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, n_chunks=1):
        super(GroupMixedConv, self).__init__()
        # in_channels -> out_channels，降维
        self.group_conv1D = GroupConv1D(in_channels, out_channels, kernel_size, n_chunks)
        # out_channels -> in_channels，升维
        self.mixed_conv = MixedConv(out_channels, in_channels, n_chunks, stride)

    def forward(self, x):
        out = self.group_conv1D(x)
        out = self.mixed_conv(out)
        return out

def split_layer(total_channels, num_groups):
    '''
    分隔channels
    :param total_channels:
    :param num_groups:
    :return:
    '''
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split

class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, bias=False):
        super(DepthwiseConv1D, self).__init__()
        padding = (kernal_size - 1) // 2
        # 深度卷积
        self.depthwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernal_size, padding=padding, stride=stride, groups=in_channels, bias=bias)

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out

class GroupConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, n_chunks=1, bias=False):
        super(GroupConv1D, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        split_out_channels = split_layer(out_channels, n_chunks)

        if n_chunks == 1:
            self.group_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2, bias=bias)
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Conv1d(self.split_in_channels[idx], split_out_channels[idx], kernel_size=kernel_size, padding=(kernel_size - 1)//2, bias=bias))

    def forward(self, x):
        if self.n_chunks == 1:
            return self.group_conv(x)
        else:
            split = torch.split(x, self.split_in_channels, dim=1)
            out = torch.cat([layer(s) for layer, s in zip(self.group_layers, split)], dim=1)
            # print('group out shape:{}'.format(out.shape))
            return out # [batch, out_channels, src_len]

class MixedConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_chunks, stride=1, bias=False):
        super(MixedConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        self.split_out_channels = split_layer(out_channels, n_chunks)

        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(DepthwiseConv1D(self.split_in_channels[idx], self.split_out_channels[idx], kernal_size=kernel_size, stride=stride, bias=bias))

    def forward(self, x):
        '''
        :param x: [batch, out_channels, src_len]
        :return:
        '''
        split = torch.split(x, self.split_in_channels, dim=1) # 对x进行分隔
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1) # 分别对分隔后的x进行深度卷积，最后拼接回原来维度
        # print('mixed out shape:{}'.format(out.shape))
        return out # [batch, out_channels, src_len]













