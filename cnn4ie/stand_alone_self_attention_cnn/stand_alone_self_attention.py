import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, kernel_size), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        '''
        :param x: [batch_size, hid_dim, src_len]
        :return:
        '''
        batch, in_channels, src_len = x.size()

        padded_x = F.pad(x, [self.padding, self.padding]) # 维度扩充, [batch, in_channels, src_len + 2 * self.padding]

        q_out = self.query_conv(x) # [batch, out_channels, src_len]
        k_out = self.key_conv(padded_x) # [batch, out_channels, src_len + 2 * self.padding]
        v_out = self.value_conv(padded_x) # [batch, out_channels, src_len + 2 * self.padding]

        # 巻
        k_out = k_out.unfold(2, self.kernel_size, self.stride) # [batch, out_channels, src_len, kernel_size]
        v_out = v_out.unfold(2, self.kernel_size, self.stride) # [batch, out_channels, src_len, kernel_size]

        # [batch, in_channels, src_len, kernel_size] 因为out_channels=2 * in_channels
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1) # (分隔长度为out_channels一半)第二维进行分离
        # [batch, out_channels, src_len, kernel_size]
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1) # 拼接

        # [batch, groups, out_channels//groups, src_len,  kernel_size]
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, src_len, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, src_len, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, src_len, 1) # [batch,groups,out_channels//groups,src_len,1]

        out = q_out * k_out # [batch, groups, out_channels//groups, src_len, kernel_size]

        out = F.softmax(out, dim=-1)
        # [batch, out_channels, src_len]
        out = torch.einsum('bncsk,bncsk -> bncs', out, v_out).view(batch, -1, src_len) # [batch, n*c, src_len]

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)