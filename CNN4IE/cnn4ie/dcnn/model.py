import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention1d(nn.Module):
    def __init__(self, in_planes, ratio, K, temperature, init_weight=True):
        '''

        :param in_planes:
        :param ratio:
        :param K:
        :param temperature:
        :param init_weight:
        '''
        super(Attention1d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratio) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        #self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('change temperature to :', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class DynamicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34, init_weight=True):
        '''

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param ratio:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param K:
        :param temperature:
        :param init_weight:
        '''
        super(DynamicConv1d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = Attention1d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x): # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height = x.size()
        x = x.view(1, -1, height, ) # 转为一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成，生成的是batch_size个卷积参数(每个参数不同)
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, )
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-1))
        return output


'''
class Attention2d(nn.Module):
    def __init__(self, in_planes, ratio, K, temperature, init_weight=True):
        super(Attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratio) # 这边是为了做一个bottleneck结构
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # conv2d初始化参数，fc是否需要
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # 直接初始化bias为0

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        pdb.set_trace()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)

class DynamicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34, init_weight=True):
        super(DynamicConv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = Attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x) # [batch_size, K]
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1) # [K, ***]
        #
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size) # 将batch_size和output_channel两个维度融合，用于后面的分组卷积，
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1) # [batch_size,out_planes] -> [b*o]
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*batch_size) # 与前面呼应
            
            #若groups本身为1，则这里用分组卷积就是把上面的batch_size*output_channel又拆分开
            #就是把input的x切分为batch_size组，kernel也分为Batch_size组，分别卷积，之后把结果concat起来
            #这样每个sample对应的都是不同的kernel
            
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
'''
