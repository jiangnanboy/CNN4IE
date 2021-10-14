import torch
import torch.nn as nn

class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MultiConv, self).__init__()
        # define the three convolutional layers with different context sizes
        # the convolutions are implemented as separable convolutions for reducing the number of model parameters
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=in_channels, kernel_size=(kernel_size, ), padding=(kernel_size-1)//2)
        self.conv1_sep = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=in_channels, kernel_size=(kernel_size+2, ), padding=(kernel_size+2-1)//2)
        self.conv2_sep = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=in_channels, kernel_size=(kernel_size+4, ), padding=(kernel_size+4-1)//2)
        self.conv3_sep = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # conv4 here is a standard convolution layer, for reshape the concatenation
        self.conv4 = nn.Conv1d(in_channels=out_channels*3, out_channels=out_channels, kernel_size=(kernel_size, ), padding=(kernel_size-1)//2)

    def forward(self, x):
        '''
        :param x: [batch_size, in_channels, src_len]
        :return:
        '''
        # [batch, out_channels, src_len]
        conv1_out = self.conv1(x)
        # print('conv1_out shape:{}'.format(conv1_out.shape))
        conv1_out = self.conv1_sep(conv1_out)
        # print('conv1_out shape:{}'.format(conv1_out.shape))
        conv1_out = self.leakyrelu1(conv1_out)
        # print('conv1_out shape:{}'.format(conv1_out.shape))

        # [batch, out_channels, src_len]
        conv2_out = self.conv2(x)
        conv2_out = self.conv2_sep(conv2_out)
        conv2_out = self.leakyrelu2(conv2_out)

        # [batch, out_channels, src_len]
        conv3_out = self.conv3(x)
        conv3_out = self.conv3_sep(conv3_out)
        conv3_out = self.leakyrelu3(conv3_out)

        # combine the outputs of convolutional layers with different context size
        x = torch.cat((conv1_out, conv2_out, conv3_out), 1) # [batch, 3 * out_channels, src_len]

        conv4_out = self.conv4(x) # [batch, out_channels, src_len]
        conv4_out = self.leakyrelu4(conv4_out)
        return conv4_out

