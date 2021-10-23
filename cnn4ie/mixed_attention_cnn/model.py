import torch
import torch.nn.functional as F
import torch.nn as nn
from cnn4ie.util import crf
from cnn4ie.mixed_attention_cnn.mixed_attention import MixedAttention


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, head_num, head_ratio, n_layers, kernel_size, dropout):
        '''
        define encoder
        :param emb_dim:
        :param hid_dim:
        :param head_num:
        :param head_ratio:
        :param n_layers:
        :param kernel_size:
        :param dropout:
        '''
        super(Encoder, self).__init__()

        # for kernel in kernel_size:
        assert kernel_size % 2 == 1, 'kernel size must be odd!'  # kernel is odd, which is convenient for PAD processing on both sides of the sequence

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(DEVICE)  # the variance of the entire network does not change significantly

        self.emb2hid = nn.Linear(emb_dim, hid_dim)  # fc: emb_dim -> hid_dim
        self.hid2emb = nn.Linear(hid_dim, emb_dim)  # fc: hid_dim -> emb_dim

        # convolution block
        self.convs = nn.ModuleList([MixedAttention(hidden_size=hid_dim,
                                                   num_attention_heads=head_num,
                                                   head_ratio=head_ratio,
                                                   conv_kernel_size=kernel_size,
                                                   dropout=dropout)
                                    for _ in range(n_layers)])  # convolution layer

        self.dropout = nn.Dropout(dropout)

        #self.BN = nn.BatchNorm1d()

    def forward(self, encoder_output):
        # encoder_output:[batch_size, src_len, emb_dim]

        # emb_dim -> hid_dim, as the input of convolution layers
        conv_input = self.emb2hid(encoder_output)  # [batch_size, src_len, hid_dim]
        # change dimension，convolve the last dimension of input
        # conv_input = conv_input.permute(0, 2, 1)  # [batch_size, hid_dim, src_len]

        # convolution block
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))  # [batch_size, src_len, hid_dim]

            #conved = self.BN(conved) # [batch_size, 2*hid_dim, src_len]

            # GLU activation function
            # conved = F.glu(conved, dim=1)  # [batch_size, hid_dim, src_len]
            # residual connection
            conved = (conved + conv_input) * self.scale  # [batch_size, src_len, hid_dim]
            # input of the next convolution layer
            conv_input = conved

        # hid_dim -> emb_dim，as the output of convolution block
        # conved = self.hid2emb(conved.permute(0, 2, 1))  # [batch_size, src_len, emb_dim]
        conved = self.hid2emb(conved)
        # residual connection，as the joint output feature of encoder
        combined = (conved + encoder_output) * self.scale  # [batch_size, src_len, emb_dim]

        return conved, combined

class MultiLayerMixedAttCNN(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, head_num, head_ratio, cnn_layers, encoder_layers, kernel_size, dropout, PAD_IDX, max_length=100, use_crf = True):
        '''
        define berc model
        :param input_dim:
        :param output_dim:
        :param emb_dim:
        :param hid_dim:
        :param head_num:
        :param head_ratio:
        :param cnn_layers:
        :param encoder_layers:
        :param kernel_size:
        :param dropout:
        :param padding_idx:
        :param max_length:
        '''
        super(MultiLayerMixedAttCNN, self).__init__()

        self.tok_embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)  # token embedding
        self.pos_embedding = nn.Embedding(max_length, emb_dim, padding_idx=PAD_IDX)  # position embedding

        self.encoder = nn.ModuleList([Encoder(emb_dim, hid_dim, head_num, head_ratio, cnn_layers, kernel_size, dropout)
                                      for _ in range(encoder_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.crf = crf.CRF(output_dim, batch_first=True)
        self.use_crf = use_crf

    def forward(self, token_tensor):
        '''
        :param token_tensor: [batch_size, src_len]
        :return:
        '''
        # token, position embedding
        tok_embedded = self.tok_embedding(token_tensor)  # [batch_size, src_len, emb_dim]
        # 构建位置tensor -> [batch_size, src_len]，位置序号从(0)开始到(src_len-1)
        position = torch.arange(0, token_tensor.shape[1]).unsqueeze(0).repeat(token_tensor.shape[0], 1).to(DEVICE)
        pos_embedded = self.pos_embedding(position.long())  # [batch_size, src_len, emb_dim]

        # token embedded + pos_embedded
        embedded = self.dropout(tok_embedded + pos_embedded)  # [batch_size, src_len, emb_dim]
        encoder_output = embedded

        # encoder block
        for i, encoder in enumerate(self.encoder):
            # encoding
            conved, encoder_output = encoder(self.dropout(encoder_output))  # [batch_size, src_len, emb_dim]

        # pooling, predict class of the entire sentence
        # encoder_output = F.avg_pool1d(encoder_output.permute(0, 2, 1), encoder_output.shape[1]).squeeze(2)  # [batch_size, emb_dim]
        # output = self.fc_out(encoder_output)  # [batch_size, output_dim]

        # fc outuput
        output = self.fc_out(encoder_output) # [batch_size, src_len, output_dim]

        if self.use_crf:
            # crf
            output = self.crf.decode(output)
        return output

    def log_likelihood(self, source, target):
        '''
        :param source: [batch_size, src_len]
        :param target: [batch_size, src_len]
        :return:
        '''
        # token, position embedding
        tok_embedded = self.tok_embedding(source)  # [batch_size, src_len, emb_dim]
        # 构建位置tensor -> [batch_size, src_len]，位置序号从(0)开始到(src_len-1)
        position = torch.arange(0, source.shape[1]).unsqueeze(0).repeat(source.shape[0], 1).to(DEVICE)
        pos_embedded = self.pos_embedding(position.long())  # [batch_size, src_len, emb_dim]

        # token embedded + pos_embedded
        embedded = self.dropout(tok_embedded + pos_embedded)  # [batch_size, src_len, emb_dim]
        encoder_output = embedded

        # encoder block
        for i, encoder in enumerate(self.encoder):
            # encoding
            conved, encoder_output = encoder(self.dropout(encoder_output))  # [batch_size, src_len, emb_dim]

        # pooling, predict class of the entire sentence
        # encoder_output = F.avg_pool1d(encoder_output.permute(0, 2, 1), encoder_output.shape[1]).squeeze(2)  # [batch_size, emb_dim]
        # output = self.fc_out(encoder_output)  # [batch_size, output_dim]

        # sequence labeling
        outputs = self.fc_out(encoder_output)  # [batch_size, src_len, output_dim]

        return -self.crf(outputs, target)
