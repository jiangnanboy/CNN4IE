import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, in_channels, kernel_size):
        super(SelfAttention, self).__init__()
        self.query, self.key, self.value = [self._conv(in_channels, out_channels, kernel_size) for out_channels in (in_channels//8, in_channels//8, in_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def _conv(self, in_channels, out_channels, kernel_size):

        return nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1)//2, bias=False)

    def forward(self, x):
        '''
        :param x: [batch, in_channels, src_len]
        :return:
        '''

        query = self.query(x) # [batch, in_channels//8, src_len]
        key = self.key(x) # [batch, in_channels//8, src_len]
        value = self.value(x) # [batch, in_channels, src_len]

        beta = F.softmax(torch.bmm(query.transpose(1, 2), key), dim=1) # # [batch, src_len, src_len]

        # out = self.gamma * torch.bmm(value, beta) + x
        out = self.gamma * torch.bmm(value, beta)
        # print('gamma value:{}'.format(self.gamma.item()))
        return out
