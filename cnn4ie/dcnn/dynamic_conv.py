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

def DynamicConv(input_size, kernel_size=1, padding_l=None, num_heads=1,
                weight_dropout=0., weight_softmax=False,
                renorm_padding=False, bias=False, conv_bias=False,
                query_size=None, in_proj=False, with_linear=False,
                glu=False, out_dim=None):

    return DynamicConv1dTBC(input_size, kernel_size=kernel_size,
                            padding_l=padding_l, num_heads=num_heads,
                            weight_dropout=weight_dropout,
                            weight_softmax=weight_softmax, bias=bias,
                            with_linear=with_linear, glu=glu, out_dim=out_dim)

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

class DynamicConv1dTBC(nn.Module):
    '''Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size) -> [src_len, batch_size, hid_dim]
        Output: TxBxC, i.e. (timesteps, batch_size, input_size) -> [src_len, batch_size, hid_dim]
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''

    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=True,
                 renorm_padding=True, bias=True, conv_bias=True,
                 query_size=None, in_proj=False, with_linear=True, glu=False, out_dim=None):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding

        if in_proj:
            self.weight_linear = Linear(self.input_size, self.input_size + num_heads * kernel_size * 1)
        else:
            self.weight_linear = Linear(self.query_size, num_heads * kernel_size * 1, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

        if with_linear:
            if glu:
                self.linear1 = Linear(input_size, input_size * 2)
                self.act = nn.GLU()
            else:
                self.linear1 = Linear(input_size, input_size)
                self.act = None
            self.linear2 = Linear(input_size, out_dim)

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size + self.num_heads * self.kernel_size

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x, query=None, unfold=None):
        '''Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        '''
        if self.linear1 is not None:
            x = self.linear1(x)
            if self.act is not None:
                x = self.act(x)
        unfold = x.size(
            0) > 512 if unfold is None else unfold  # use unfold mode as default for long sequence to save memory

        assert query is None or not self.in_proj

        if query is None:
            query = x
        if unfold:
            output = self._forward_unfolded(x, query)
        else:
            output = self._forward_expanded(x, query)

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)
        if self.linear2 is not None:
            output = self.linear2(output) # [src_len, batch_size, out_dim]
        return output

    def _forward_unfolded(self, x, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H * K).contiguous().view(T * B * H, -1)
        else:
            weight = self.weight_linear(query).view(T * B * H, -1)


        padding_l = self.padding_l
        if K > T and padding_l == K - 1:
            weight = weight.narrow(1, K - T, T)
            K, padding_l = T, T - 1
        # unfold the input: T x B x C --> T' x B x C x K
        x_unfold = unfold1d(x, K, padding_l, 0)
        x_unfold = x_unfold.view(T * B * H, R, K)

        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, K)

        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)

        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)

        output = torch.bmm(x_unfold, weight.unsqueeze(2))  # T*B*H x R x 1
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, query):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        '''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H * K).contiguous().view(T * B * H, -1)
        else:
            weight = self.weight_linear(query).view(T * B * H, -1)

        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B * H, K).transpose(0, 1)

        x = x.view(T, B * H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            # turn the convolution filters into band matrices
            weight_expanded = weight.new(B * H, T, T + K - 1).fill_(float('-inf'))
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            # normalize the weight over valid positions like self-attention
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training, inplace=False)
        else:
            P = self.padding_l
            # For efficieny, we cut the kernel size and reduce the padding when the kernel is larger than the length
            if K > T and P == K - 1:
                weight = weight.narrow(2, K - T, T)
                K, P = T, T - 1
            # turn the convolution filters into band matrices
            weight_expanded = weight.new_zeros(B * H, T, T + K - 1, requires_grad=False)
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)  # B*H x T x T
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output