import torch.utils.data
from torch.nn import functional as F

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple


class _ConvNd(Module): # 定义通用卷积神经网络模块，所有卷积层的基类。

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, transposed, output_padding, groups, bias, weight=None):
        """
        初始化_ConvNd类实例。

        Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int 或 tuple): 卷积核的大小。
        stride (int 或 tuple): 卷积步长。
        padding (int 或 tuple): 输入的零填充大小。
        dilation (int 或 tuple): 卷积核元素之间的间距。
        transposed (bool): 是否进行转置卷积。
        output_padding (int 或 tuple): 输出填充大小。
        groups (int): 将输入和输出通道分为多少个组。
        bias (bool): 是否添加偏置项。
        weight (torch.Tensor, 可选): 卷积核权重，默认值为None。

        Raises:
        ValueError: 如果输入通道数或输出通道数不能被分组数整除。

        """
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(weight)

    def reset_parameters(self, weight): # 参数初始化方法
        if weight == None: # 如果没有提供权重参数，使用均匀分布初始化权重
            n = self.in_channels
            for k in self.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n) # 计算标准差根号n
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weight.data = torch.FloatTensor(weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv) # 均匀分布初始化偏置

    def __repr__(self): # 打印对象方法，创建实例后，打印会调用该方法，比如 conv1d = Conv1d(...)，print(conv1d) 会调用这个方法
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv1d(_ConvNd):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='VALID', dilation=1, groups=1, bias=True, weight=None):
        kernel_size = _single(kernel_size)# 转为元组形式
        stride = _single(stride) 
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, weight)

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                       self.groups)


# custom con2d, because pytorch don't have "padding='same'" option.
def conv1d_same_padding(input, weight, bias=None, stride=1, padding='VALID', dilation=1, groups=1):
    def check_format(*argv):
        argv_format = []

        for i in range(len(argv)):
            if type(argv[i]) is int:
                argv_format.append((argv[i], argv[i]))
            elif hasattr(argv[i], "__getitem__"):
                argv_format.append(tuple(argv[i]))
            else:
                raise TypeError('all input should be int or list-type, now is {}'.format(argv[i]))

        return argv_format

    stride, dilation = check_format(stride, dilation)
    
    if padding == 'SAME':
        padding = 0

        input_rows = input.size(2)
        filter_rows = weight.size(2)
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = padding_rows % 2

        # input_cols = input.size(3)
        # filter_cols = weight.size(3)
        # out_cols = (input_cols + stride[1] - 1) // stride[1]
        # padding_cols = max(0, (out_cols - 1) * stride[1] +
        #                    (filter_cols - 1) * dilation[1] + 1 - input_cols)
        # cols_odd = padding_cols % 2
        input = pad(input, [padding_rows // 2, padding_rows // 2 + int(rows_odd)])
    elif padding == 'VALID':
        padding = 0

    elif type(padding) != int:
        raise ValueError('Padding should be SAME, VALID or specific integer, but not {}.'.format(padding))

    return F.conv1d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


class _MaxPoolNd(Module):

    def __init__(self, kernel_size, stride=None, padding='VALID', dilation=1,
                 return_indices=False, ceil_mode=False):
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class MaxPool1d(_MaxPoolNd):

    def forward(self, input):
        return maxpool1d_same_padding(input, self.kernel_size, self.stride, self.padding,
                                      self.dilation, self.return_indices, self.ceil_mode)

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


def maxpool1d_same_padding(input, kernel_size, stride=None, padding='VALID', dilation=1, return_indices=False, ceil_mode=False):

    if stride is None:
        _stride, dilation = [kernel_size], [dilation]
    else:
        _stride, dilation = [stride], [dilation]

    if padding == 'SAME':
        padding = 0

        input_rows = input.size(2)
        filter_rows = kernel_size
        out_rows = (input_rows + _stride[0] - 1) // _stride[0]
        padding_rows = max(0, (out_rows - 1) * _stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = padding_rows % 2

        # input_cols = input.size(3)
        # filter_cols = weight.size(3)
        # out_cols = (input_cols + _stride[1] - 1) // _stride[1]
        # padding_cols = max(0, (out_cols - 1) * _stride[1] +
        #                    (filter_cols - 1) * dilation[1] + 1 - input_cols)
        # cols_odd = padding_cols % 2
        input = pad(input, [padding_rows // 2, padding_rows // 2 + int(rows_odd)])

    elif padding == 'VALID':
        padding = 0

    elif type(padding) != int:
        raise ValueError('Padding should be SAME, VALID or specific integer, but not {}.'.format(padding))

    return F.max_pool1d(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)