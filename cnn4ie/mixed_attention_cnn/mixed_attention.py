import torch
import torch.nn as nn
import math

class SeparableConv1D(nn.Module):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, input_filters, output_filters, kernel_size):
        super(SeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            padding=kernel_size // 2,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))

        self.depthwise.weight.data.normal_(mean=0.0, std=0.01)
        self.pointwise.weight.data.normal_(mean=0.0, std=0.01)

    def forward(self, hidden_states):
        x = self.depthwise(hidden_states)
        x = self.pointwise(x)
        x += self.bias
        return x

class MixedAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, head_ratio, conv_kernel_size, dropout):
        super(MixedAttention, self).__init__()

        assert hidden_size % num_attention_heads == 0, f'The hidden size ({hidden_size}) is not a multiple of the number of attention'

        new_num_attention_heads = num_attention_heads // head_ratio
        if new_num_attention_heads < 1:
            self.head_ratio = num_attention_heads
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads
            self.head_ratio = head_ratio

        self.conv_kernel_size = conv_kernel_size

        assert hidden_size % num_attention_heads == 0, f'The hidden size ({hidden_size}) is not a multiple of the number of attention'

        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.key_conv_attn_layer = SeparableConv1D(hidden_size, self.all_head_size, self.conv_kernel_size)
        self.conv_kernel_layer = nn.Linear(self.all_head_size, self.num_attention_heads * self.conv_kernel_size)
        self.conv_out_layer = nn.Linear(hidden_size, self.all_head_size)

        self.unfold = nn.Unfold(kernel_size=[self.conv_kernel_size, 1], padding=[int((self.conv_kernel_size - 1) / 2), 0])

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        '''
        :param x: [batch_size, src_len, hid_dim]
        :return:
        '''
        mixed_query_layer = self.query(x)
        batch_size = x.size(0)

        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        mixed_key_conv_attn_layer = self.key_conv_attn_layer(x.transpose(1, 2))
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)

        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)

        conv_out_layer = self.conv_out_layer(x)
        conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        conv_out_layer = nn.functional.unfold(
            conv_out_layer,
            kernel_size=[self.conv_kernel_size, 1],
            dilation=1,
            padding=[(self.conv_kernel_size - 1) // 2, 0],
            stride=1,
        )
        conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
            batch_size, -1, self.all_head_size, self.conv_kernel_size
        )
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        context_layer = torch.cat([context_layer, conv_out], 2)

        new_context_layer_shape = context_layer.size()[:-2] + (self.head_ratio * self.all_head_size,)
        outputs = context_layer.view(*new_context_layer_shape) # [batch_size, src_len, hid_dim]

        return outputs
