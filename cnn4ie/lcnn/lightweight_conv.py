import torch
import torch.nn as nn
import torch.nn.functional as F

def unfold1d(x, kernel_size, padding_l, pad_value=0):
    '''
    unfold T x B x C to T x B x C x K
    :param x: [src_len, batch_size, hid_dim]
    :param kernel_size:
    :param padding_l:
    :param pad_value:
    :return:
    '''
    if kernel_size > 1:
        T, B, C = x.size()
        '''
        F.pad() 维度扩充
        
        1.x：需要扩充的tensor
        2.(0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l)：前4个参数完成了在高和宽维度上的扩张，后两个参数则完成了对通道维度上的扩充
        (左边填充数，右边填充数，上边填充数，下边填充数，前边填充数，后边填充数)
        3.value:扩充时指定补充值
        '''
        x = F.pad(
            x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value
        )
        '''
        as_strided() 根据现有tensor以及给定的步长来创建一个视图
        
        1.size：指定了生成的视图的大小，需要为一个矩阵（当然此矩阵大小可以大于原矩阵，但是也有限制），可以是tensor或者list等等。
        2.stride：输出tensor的步长，根据原矩阵和步长生成了新矩阵。
        '''
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))
    else:
        x = x.unsqueeze(3)
    return x

class LightweightConv1dTBC(nn.Module):
    '''Lightweight Convolution assuming the input is TxBxC
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias

    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size) -> [src_len, batch_size, hid_dim]
        Output: TxBxC, i.e. (timesteps, batch_size, input_size) -> [src_len, batch_size, hid_dim]
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''

    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=True, bias=True, with_linear=True, out_dim=None):
        super().__init__()
        self.embed_dim = input_size
        out_dim = input_size if out_dim is None else out_dim
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size)) # [num_heads, 1, kernel_size]
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.linear1 = self.linear2 = None
        if with_linear:
            self.linear1 = Linear(input_size, input_size)
            self.linear2 = Linear(input_size, out_dim)

        self.reset_parameters() # 初始化 weight

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, unfold=False):
        '''
        Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            unfold: unfold the input or not. If not, we use the matrix trick instead
        :param x: [src_len, batch_size, hid_dim]
        :param unfold:
        :return:
        '''
        if self.linear1 is not None:
            x = self.linear1(x)

        if unfold:
            # [src_len, batch_size, hid_dim]
            output = self._forward_unfolded(x)
        else:
            # [src_len, batch_size, hid_dim]
            output = self._forward_expanded(x)

        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1) # bias: [1, 1, hid_dim]

        if self.linear2 is not None:
            output = self.linear2(output) # [src_len, batch_size, out_dim]
        return output

    def _forward_unfolded(self, x):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size() # [src_len, batch_size, hid_dim]
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight.view(H, K) # [num_heads, 1, kernel_size] -> [num_heads, kernel_size]

        # unfold the input: T x B x C --> T' x B x C x K [src_len, batch_size, hid_dim, kernel_size]
        x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)
        x_unfold = x_unfold.view(T * B * H, R, K) # [src_len * batch_size * num_heads, hid_dim//num_heads, kernel_size]

        if self.weight_softmax:
            # [num_heads, kernel_size]
            weight = F.softmax(weight, dim=1, dtype=torch.float32).type_as(weight)

        # [src_len * batch_size * num_heads, kernel_size, 1]
        weight = weight.view(1, H, K).expand(T * B, H, K).contiguous().view(T * B * H, K, 1)

        weight = F.dropout(weight, self.weight_dropout)
        # [src_len * batch_size * num_heads, hid_dim//num_heads, kernel_size] * [src_len * batch_size * num_heads, kernel_size, 1]
        output = torch.bmm(x_unfold, weight)  # T*B*H x R x 1 [src_len * batch_size * num_heads, hid_dim//num_heads, 1]
        output = output.view(T, B, C) # [src_len, batch_size, hid_dim]
        return output

    def _forward_expanded(self, x):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        '''
        T, B, C = x.size() # [src_len, batch_size, hid_dim]
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight.view(H, K) # [num_heads, 1, kernel_size] -> [num_heads, kernel_size]
        if self.weight_softmax:
            weight = F.softmax(weight, dim=1, dtype=torch.float32).type_as(weight)
        weight = weight.view(1, H, K).expand(T * B, H, K).contiguous() # [src_len * batch_size, num_heads, kernel_size]
        weight = weight.view(T, B * H, K).transpose(0, 1) # [batch_size * num_heads, src_len, kernel_size]

        x = x.view(T, B * H, R).transpose(0, 1) # [batch_size * num_heads, src_len, hid_dim//num_heads]
        P = self.padding_l
        if K > T and P == K - 1:
            weight = weight.narrow(2, K - T, T)
            K, P = T, T - 1
        # turn the convolution filters into band matrices
        weight_expanded = weight.new_zeros(B * H, T, T + K - 1, requires_grad=False)
        weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, P, T)
        weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training)

        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C) # [src_len, batch_size, hid_dim]
        return output

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m