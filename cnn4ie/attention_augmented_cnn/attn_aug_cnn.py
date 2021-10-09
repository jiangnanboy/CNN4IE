import torch
import torch.nn as nn
import torch.nn.functional as F

class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, padding, stride=1):
        '''
        Nh、dv 和 dk 分别是指多头注意（multihead-attention / MHA）中头（head）的数量、值的深度、查询和键值的深度。
        进一步假设 Nh 均匀地划分 dv 和 dk，并用dvh和dkh分别表示每个注意头的值和查询/键值的深度。
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param dk:
        :param dv:
        :param Nh:
        :param stride:
        '''
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.stride = stride
        self.padding = padding

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, padding=self.padding)

        self.qkv_conv = nn.Conv1d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.attn_out = nn.Conv1d(self.dv, self.dv, kernel_size=1, stride=1)

    def forward(self, x):
        # Input x -> [batch_size, hid_dim, src_len]

        # conv_out -> [batch_size, out_channels, src_len]
        conv_out = self.conv_out(x)
        batch, _, src_len = conv_out.size()

        # flat_q, flat_k, flat_v
        # [batch_size, Nh, dvh or dkh, src_len]
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # [batch_size, Nh, dv or dk, src_len]
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k) # [batch, Nh, src_len, src_len]

        weights = F.softmax(logits, dim=-1) # [batch, Nh, src_len, src_len]

        # attn_out
        # [batch, Nh, src_len, dvh]
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3)) # [batch, Nh, src_len, src_len] * [batch, Nh, src_len, dvh]
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, src_len)) # [batch, Nh, dvh, src_len]
        # combine_heads_1d
        # [batch, out_channels, src_len] -> [batch, Nh * dv, src_len]
        attn_out = self.combine_heads_1d(attn_out)
        attn_out = self.attn_out(attn_out)
        result_out = torch.cat((conv_out, attn_out), dim=1)
        # conv_out -> [batch, hid_dim, src_len]
        # attn_out -> [batch, hid_dim, src_len]
        # result_out -> [batch, 2 * hid_dim, src_len]

        # print('conv_out shape:{}'.format(conv_out.shape))
        # print('attn_out shape:{}'.format(attn_out.shape))
        # print('result_out shape:{}'.format(result_out.shape))


        return result_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        '''
        x -> [batch_size, hid_dim, src_len]
        :param x:
        :param dk:
        :param dv:
        :param Nh:
        :return:
        '''
        qkv = self.qkv_conv(x)
        # [batch_size, 2 * self.dk + self.dv, src_len]
        # N, _, H, W = qkv.size()
        N, _, src_len = qkv.size()
        # q:[batch_size, dk, src_len]
        # k:[batch_size, dk, src_len]
        # v:[batch_size, kv, src_len]
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_1d(q, Nh) # [batch, Nh, dk // Nh, src_len]
        k = self.split_heads_1d(k, Nh) # [batch, Nh, dk // Nh, src_len]
        v = self.split_heads_1d(v, Nh) # [batch, Nh, dv // Nh, src_len]
        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, src_len)) # [batch, Nh, dk // Nh, src_len]

        flat_k = torch.reshape(k, (N, Nh, dk // Nh, src_len)) # [batch, Nh, dk // Nh, src_len]
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, src_len)) # [batch, Nh, dv // Nh, src_len]

        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_1d(self, x, Nh):
        '''
        :param x: [batch_size, dk or dv , src_len]
        :param Nh:
        :return:
        '''
        # batch, channels, height, width = x.size()
        batch, channels, src_len = x.size()
        ret_shape = (batch, Nh, channels // Nh, src_len)
        split = torch.reshape(x, ret_shape) # batch, Nh, channels // Nh, src_len
        return split

    def combine_heads_1d(self, x):
        '''
        :param x: [batch, Nh, dvh, src_len]
        :return:
        '''
        batch, Nh, dv, src_len = x.size()
        ret_shape = (batch, Nh * dv, src_len)
        return torch.reshape(x, ret_shape)

