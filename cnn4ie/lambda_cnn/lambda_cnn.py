import torch
import torch.nn as nn

class LambdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, heads=4, k=16, u=1):
        '''
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param padding:
        :param heads: number of heads, for multi-query
        :param k:key dimension
        :param u:'intra-depth' dimension
        '''
        super(LambdaConv, self).__init__()
        self.kk = k
        self.uu = u
        self.vv = out_channels // heads
        self.heads = heads

        self.queries = nn.Sequential(
            nn.Conv1d(in_channels, k * heads, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(k * heads)
        )
        self.keys = nn.Sequential(
            nn.Conv1d(in_channels, k * u, kernel_size=kernel_size, padding= padding, bias=False),
        )
        self.values = nn.Sequential(
            nn.Conv1d(in_channels, self.vv * u, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(self.vv * u)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        '''
        :param x: [batch_size, hid_dim, src_len]
        :return:
        '''
        # batch, C, w, h = x.size()
        batch, hid_dim, src_len = x.size()

        queries = self.queries(x) # [batch,hid_dim * heads,src_Len]
        queries = queries.view(batch, self.heads, self.kk, src_len) # [b, heads, k // heads, src_len]
        softmax = self.softmax(self.keys(x).view(batch, self.kk, self.uu, src_len)) # [b, k, uu, src_len]
        values = self.values(x).view(batch, self.vv, self.uu, src_len) # [b, v, uu, src_len]

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values) # [batch, k, v]
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c) # [batch, heads, v, src_len]

        lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values) # [batch, kk, v, src_len]
        y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p) # [batch, heads, v, src_len]

        result_out = y_c + y_p # [batch, heads, v, src_len]
        result_out = result_out.contiguous().view(batch, -1, src_len) # [batch, heads*v, src_len] -> [batch, out_channels, src_len]

        # y_c -> [batch, heads, out_channels // heads, src_len]
        # y_p -> [batch, heads, out_channels // heads, src_len]
        # result_out -> [batch, out_channels ,src_len]

        return result_out