"""Define basic models and translate some torchvision stuff."""
"""Stuff from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py."""
import torch
import torchvision
import torch.nn as nn

from torchvision.models.resnet import Bottleneck
from .revnet import iRevNet
from .densenet import _DenseNet, _Bottleneck

from collections import OrderedDict
import numpy as np
from ..utils import set_random_seed
import math
import torch.nn.functional as F


def construct_model(model, num_classes=10, seed=None, num_channels=3, modelkey=None, use_rp=False, rp_ratio=0.25, rp_eps=0.2):
    """Return various models."""
    if modelkey is None:
        if seed is None:
            model_init_seed = np.random.randint(0, 2**32 - 10)
        else:
            model_init_seed = seed
    else:
        model_init_seed = modelkey
    set_random_seed(model_init_seed)

    if model in ['ConvNet', 'ConvNet64']:
        model = ConvNet(width=64, num_channels=num_channels, num_classes=num_classes, 
                       use_rp=use_rp, rp_ratio=rp_ratio, rp_eps=rp_eps)
    elif model == 'ConvNet8':
        model = ConvNet(width=8, num_channels=num_channels, num_classes=num_classes,
                       use_rp=use_rp, rp_ratio=rp_ratio, rp_eps=rp_eps)
    elif model == 'ConvNet16':
        model = ConvNet(width=16, num_channels=num_channels, num_classes=num_classes,
                       use_rp=use_rp, rp_ratio=rp_ratio, rp_eps=rp_eps)
    elif model == 'ConvNet32':
        model = ConvNet(width=32, num_channels=num_channels, num_classes=num_classes,
                       use_rp=use_rp, rp_ratio=rp_ratio, rp_eps=rp_eps)
    elif model == 'ConvNet128':
        model = ConvNet(width=128, num_channels=num_channels, num_classes=num_classes,
                       use_rp=use_rp, rp_ratio=rp_ratio, rp_eps=rp_eps)
    
    elif model == 'BeyondInferringMNIST':
        model = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)),
            ('relu0', torch.nn.LeakyReLU()),
            ('conv2', torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            ('relu1', torch.nn.LeakyReLU()),
            ('conv3', torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            ('relu2', torch.nn.LeakyReLU()),
            ('conv4', torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            ('relu3', torch.nn.LeakyReLU()),
            ('flatt', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(12544, 12544)),
            ('relu4', torch.nn.LeakyReLU()),
            ('linear1', torch.nn.Linear(12544, 10)),
            ('softmax', torch.nn.Softmax(dim=1))
        ]))
    elif model == 'BeyondInferringCifar':
        model = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)),
            ('relu0', torch.nn.LeakyReLU()),
            ('conv2', torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            ('relu1', torch.nn.LeakyReLU()),
            ('conv3', torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            ('relu2', torch.nn.LeakyReLU()),
            ('conv4', torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            ('relu3', torch.nn.LeakyReLU()),
            ('flatt', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(12544, 12544)),
            ('relu4', torch.nn.LeakyReLU()),
            ('linear1', torch.nn.Linear(12544, 10)),
            ('softmax', torch.nn.Softmax(dim=1))
        ]))
    elif model == 'MLP':
        width = 1024
        model = torch.nn.Sequential(OrderedDict([
            ('flatten', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(3072, width)),
            ('relu0', torch.nn.ReLU()),
            ('linear1', torch.nn.Linear(width, width)),
            ('relu1', torch.nn.ReLU()),
            ('linear2', torch.nn.Linear(width, width)),
            ('relu2', torch.nn.ReLU()),
            ('linear3', torch.nn.Linear(width, num_classes))]))
    elif model == 'TwoLP':
        width = 2048
        model = torch.nn.Sequential(OrderedDict([
            ('flatten', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(3072, width)),
            ('relu0', torch.nn.ReLU()),
            ('linear3', torch.nn.Linear(width, num_classes))]))
    elif model == 'ResNet20':
        model = ResNet(RandomProjectionBasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16)
    elif model == 'ResNet20-nostride':
        model = ResNet(RandomProjectionBasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16,
                       strides=[1, 1, 1, 1])
    elif model == 'ResNet20-10':
        model = ResNet(RandomProjectionBasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet20-4':
        model = ResNet(RandomProjectionBasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 4)
    elif model == 'ResNet20-4-unpooled':
        model = ResNet(RandomProjectionBasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 4,
                       pool='max')
    elif model == 'ResNet28-10':
        model = ResNet(RandomProjectionBasicBlock, [4, 4, 4], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet32':
        model = ResNet(RandomProjectionBasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16)
    elif model == 'ResNet32-10':
        model = ResNet(RandomProjectionBasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet44':
        model = ResNet(RandomProjectionBasicBlock, [7, 7, 7], num_classes=num_classes, base_width=16)
    elif model == 'ResNet56':
        model = ResNet(RandomProjectionBasicBlock, [9, 9, 9], num_classes=num_classes, base_width=16)
    elif model == 'ResNet110':
        model = ResNet(RandomProjectionBasicBlock, [18, 18, 18], num_classes=num_classes, base_width=16)
    elif model == 'ResNet18':
        model = ResNet(RandomProjectionBasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64)
    elif model == 'ResNet34':
        model = ResNet(RandomProjectionBasicBlock, [3, 4, 6, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet50':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet50-2':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64 * 2)
    elif model == 'ResNet101':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet152':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, base_width=64)
    elif model == 'MobileNet':
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # cifar adaptation, cf.https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        model = torchvision.models.MobileNetV2(num_classes=num_classes,
                                               inverted_residual_setting=inverted_residual_setting,
                                               width_mult=1.0)
        model.features[0] = torchvision.models.mobilenet.ConvBNReLU(num_channels, 32, stride=1)  # this is fixed to width=1
    elif model == 'MNASNet':
        model = torchvision.models.MNASNet(1.0, num_classes=num_classes, dropout=0.2)
    elif model == 'DenseNet121':
        model = torchvision.models.DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                                            num_init_features=64, bn_size=4, drop_rate=0, num_classes=num_classes,
                                            memory_efficient=False)
    elif model == 'DenseNet40':
        model = _DenseNet(_Bottleneck, [6, 6, 6, 0], growth_rate=12, num_classes=num_classes)
    elif model == 'DenseNet40-4':
        model = _DenseNet(_Bottleneck, [6, 6, 6, 0], growth_rate=12 * 4, num_classes=num_classes)
    elif model == 'SRNet3':
        model = SRNet(upscale_factor=3, num_channels=num_channels)
    elif model == 'SRNet1':
        model = SRNet(upscale_factor=1, num_channels=num_channels)
    elif model == 'iRevNet':
        if num_classes <= 100:
            in_shape = [num_channels, 32, 32]  # only for cifar right now
            model = iRevNet(nBlocks=[18, 18, 18], nStrides=[1, 2, 2],
                            nChannels=[16, 64, 256], nClasses=num_classes,
                            init_ds=0, dropout_rate=0.1, affineBN=True,
                            in_shape=in_shape, mult=4)
        else:
            in_shape = [3, 224, 224]  # only for imagenet
            model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                            nChannels=[24, 96, 384, 1536], nClasses=num_classes,
                            init_ds=2, dropout_rate=0.1, affineBN=True,
                            in_shape=in_shape, mult=4)
    elif model == 'LeNetZhu':
        model = LeNetZhu(num_channels=num_channels, num_classes=num_classes)
    elif model == 'LeNetZhuMNIST':
        model = LeNetZhuMNIST(num_channels=1, num_classes=num_classes, 
                             use_rp=use_rp, rp_ratio=rp_ratio, epsilon=rp_eps)
    else:
        raise NotImplementedError('Model not implemented.')

    print(f'Model initialized with random key {model_init_seed}.')
    return model, model_init_seed


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
## RPF 每次前向传播就重新采样权重
# class RandomProjectionConv(nn.Module):
#     """通用的随机投影卷积层（每次前向传播重新采样，均值=1，方差=1/r^2）"""
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, 
#                  rp_ratio=0.25, bias=True,epsilon=0.2,use_rp=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # 统一处理 kernel_size 和 padding
#         self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
#         self.stride = stride
#         self.padding = padding if padding is not None else self.kernel_size[0] // 2
#         self.use_bias = bias
#         self.epsilon = epsilon  # 用于控制随机投影的尺度
#         self.use_rp = use_rp

#         # 计算随机投影和可训练滤波器的数量
#         if use_rp:
#             self.rp_channels = int(out_channels * rp_ratio)
#             self.train_channels = out_channels - self.rp_channels
#         else:
#             self.rp_channels = 0
#             self.train_channels = out_channels
#         print(f"RandomProjectionConv: {self.train_channels} trainable + {self.rp_channels} random projection filters")

#         # 可训练卷积分支
#         if self.train_channels > 0:
#             self.train_conv = nn.Conv2d(
#                 in_channels, self.train_channels, self.kernel_size,
#                 stride=stride, padding=self.padding, bias=bias
#             )

#         # 仅存储形状信息，前向再采样
#         if self.rp_channels > 0:
#             self.rp_weight_shape = (self.rp_channels, in_channels, *self.kernel_size)
#             if bias:
#                 self.rp_bias_shape = (self.rp_channels,)
#             else:
#                 self.rp_bias_shape = None

#     def forward(self, x):
#         outputs = []
#         if self.train_channels > 0:
#             outputs.append(self.train_conv(x))

#         if self.rp_channels > 0:
#             # 拉普拉斯分布参数 b（尺度），可以通过 rp_eps 传入
#             b = self.epsilon  # 比如 0.2

#             # 从 Laplace(loc=1, scale=b) 采样权重
#             # 方法1：用 torch.distributions
#             dist = torch.distributions.Laplace(loc=1.0, scale=b)
#             rp_weight = dist.sample(self.rp_weight_shape).to(x.device)

#             # 如果需要偏置，同理：
#             rp_bias = None
#             if self.use_bias and self.rp_bias_shape is not None:
#                 rp_bias = dist.sample(self.rp_bias_shape).to(x.device)

#             # 做卷积
#             rp_out = F.conv2d(x, rp_weight, bias=rp_bias,
#                               stride=self.stride, padding=self.padding)
#             outputs.append(rp_out)

#         if len(outputs) == 1:
#             return outputs[0]
#         else:
#             return torch.cat(outputs, dim=1)


class RandomProjectionConv(nn.Module):
    """通用的随机投影卷积层（随机投影滤波器仅采样一次，均值=1，方差=1/r^2）"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, 
                 rp_ratio=0.25, bias=True, epsilon=0.2, use_rp=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 统一处理 kernel_size 和 padding
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else self.kernel_size[0] // 2
        self.use_bias = bias
        self.epsilon = epsilon  # 用于控制随机投影的尺度
        self.use_rp = use_rp

        # 计算随机投影和可训练滤波器的数量
        if use_rp:
            self.rp_channels = int(out_channels * rp_ratio)
            self.train_channels = out_channels - self.rp_channels
        else:
            self.rp_channels = 0
            self.train_channels = out_channels
        print(f"RandomProjectionConv: {self.train_channels} trainable + {self.rp_channels} random projection filters")

        # 可训练卷积分支
        if self.train_channels > 0:
            self.train_conv = nn.Conv2d(
                in_channels, self.train_channels, self.kernel_size,
                stride=stride, padding=self.padding, bias=bias
            )

        # 随机投影滤波器 - 初始化时采样一次
        if self.rp_channels > 0:
            # 拉普拉斯分布参数 b（尺度）
            b = self.epsilon  # 比如 0.2
            
            # 从 Laplace(loc=1, scale=b) 采样权重，只采样一次
            dist = torch.distributions.Laplace(loc=1.0, scale=b)
            rp_weight_shape = (self.rp_channels, in_channels, *self.kernel_size)
            rp_weight = dist.sample(rp_weight_shape)
            
            # 将随机投影权重注册为缓冲区（不参与训练，但会随模型保存/加载）
            self.register_buffer('rp_weight', rp_weight)
            
            # 如果需要偏置，同理
            if bias:
                rp_bias_shape = (self.rp_channels,)
                rp_bias = dist.sample(rp_bias_shape)
                self.register_buffer('rp_bias', rp_bias)
            else:
                self.register_buffer('rp_bias', None)

    def forward(self, x):
        outputs = []
        if self.train_channels > 0:
            outputs.append(self.train_conv(x))

        if self.rp_channels > 0:
            # 使用预先采样的随机投影权重
            rp_out = F.conv2d(x, self.rp_weight, bias=self.rp_bias,
                              stride=self.stride, padding=self.padding)
            outputs.append(rp_out)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return torch.cat(outputs, dim=1)





class ConvNet(torch.nn.Module):
    """ConvNetBN with optional random projection."""

    def __init__(self, width=32, num_classes=10, num_channels=3, use_rp=False, rp_ratio=0.25, rp_eps=0.2):
        """Init with width and num classes."""
        super().__init__()
        self.use_rp = use_rp
        
        if use_rp:
            # 使用随机投影卷积层
            layers = [
                ('conv0', RandomProjectionConv(num_channels, 1 * width, kernel_size=3, padding=1, 
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh0', torch.nn.Tanh()),

                ('conv1', RandomProjectionConv(1 * width, 2 * width, kernel_size=3, padding=1,
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh1', torch.nn.Tanh()),

                ('conv2', RandomProjectionConv(2 * width, 2 * width, kernel_size=3, padding=1,
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh2', torch.nn.Tanh()),

                ('conv3', RandomProjectionConv(2 * width, 4 * width, kernel_size=3, padding=1,
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh3', torch.nn.Tanh()),

                ('conv4', RandomProjectionConv(4 * width, 4 * width, kernel_size=3, padding=1,
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh4', torch.nn.Tanh()),

                ('conv5', RandomProjectionConv(4 * width, 4 * width, kernel_size=3, padding=1,
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh5', torch.nn.Tanh()),

                ('pool0', torch.nn.MaxPool2d(3)),

                ('conv6', RandomProjectionConv(4 * width, 4 * width, kernel_size=3, padding=1,
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh6', torch.nn.Tanh()),

                ('conv7', RandomProjectionConv(4 * width, 4 * width, kernel_size=3, padding=1,
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh7', torch.nn.Tanh()),

                ('conv8', RandomProjectionConv(4 * width, 4 * width, kernel_size=3, padding=1,
                                             rp_ratio=rp_ratio, epsilon=rp_eps, use_rp=use_rp)),
                ('tanh8', torch.nn.Tanh()),

                ('pool1', torch.nn.MaxPool2d(3)),
                ('flatten', torch.nn.Flatten()),
                ('linear', torch.nn.Linear(36 * width, num_classes))
            ]
        else:
            # 原始实现
            layers = [
                ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                ('tanh0', torch.nn.Tanh()),

                ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                ('tanh1', torch.nn.Tanh()),

                ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
                ('tanh2', torch.nn.Tanh()),

                ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
                ('tanh3', torch.nn.Tanh()),

                ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('tanh4', torch.nn.Tanh()),

                ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('tanh5', torch.nn.Tanh()),

                ('pool0', torch.nn.MaxPool2d(3)),

                ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('tanh6', torch.nn.Tanh()),

                ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('tanh7', torch.nn.Tanh()),

                ('conv8', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('tanh8', torch.nn.Tanh()),

                ('pool1', torch.nn.MaxPool2d(3)),
                ('flatten', torch.nn.Flatten()),
                ('linear', torch.nn.Linear(36 * width, num_classes))
            ]

        self.model = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, input):
        return self.model(input)


class LeNetZhuMNIST(nn.Module):
    """LeNet variant with optional random projection filters."""

    def __init__(self, num_classes=10, num_channels=3, use_rp=True, rp_ratio=0.25, epsilon=0.2):
        """3-Layer sigmoid Conv with large linear layer and optional random projection."""
        super().__init__()
        self.use_rp = use_rp
        act = nn.Sigmoid
        
        if use_rp:
            # 使用随机投影卷积层
            self.conv1 = RandomProjectionConv(
                in_channels=num_channels,
                out_channels=12,
                kernel_size=5,
                stride=2,
                padding=5//2,
                rp_ratio=rp_ratio,
                epsilon=epsilon,
                use_rp=use_rp
            )
            
            self.conv2 = RandomProjectionConv(
                in_channels=12,
                out_channels=12,
                kernel_size=5,
                stride=2,
                padding=5//2,
                rp_ratio=rp_ratio,
                epsilon=epsilon,
                use_rp=use_rp
            )
            
            self.conv3 = RandomProjectionConv(
                in_channels=12,
                out_channels=12,
                kernel_size=5,
                stride=1,
                padding=5//2,
                rp_ratio=rp_ratio,
                epsilon=epsilon,
                use_rp=use_rp
            )
            
            self.act1 = act()
            self.act2 = act()
            self.act3 = act()
        else:
            # 原始实现
            self.body = nn.Sequential(
                nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
                act(),
            )
            
        self.fc = nn.Sequential(
            nn.Linear(588, num_classes)
        )
        
        # 初始化权重
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias") and m.bias is not None and isinstance(m.bias, nn.Parameter):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        if self.use_rp:
            # 使用随机投影层的前向传播
            out = self.conv1(x)
            out = self.act1(out)
            out = self.conv2(out)
            out = self.act2(out)
            out = self.conv3(out)
            out = self.act3(out)
        else:
            # 原始前向传播
            out = self.body(x)
            
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=[1, 2, 2, 2], pool='avg'):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class LeNetZhu(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SRNet(nn.Module):
    """Super Resolution Network for upscaling images."""
    
    def __init__(self, upscale_factor, num_channels=3):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # Feature extraction
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        
        # Upscaling
        if upscale_factor > 1:
            self.conv3 = nn.Conv2d(32, num_channels * (upscale_factor ** 2), kernel_size=5, padding=2)
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        else:
            self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        
        if self.upscale_factor > 1:
            out = self.pixel_shuffle(out)
        
        return self.tanh(out)


# 随机投影版本的ResNet基础块
class RandomProjectionBasicBlock(nn.Module):
    """BasicBlock with random projection support"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_rp=False, rp_ratio=0.25, rp_eps=0.2):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if use_rp:
            self.conv1 = RandomProjectionConv(inplanes, planes, kernel_size=3, stride=stride, padding=1, 
                                            rp_ratio=rp_ratio, epsilon=rp_eps, bias=False, use_rp=use_rp)
            self.conv2 = RandomProjectionConv(planes, planes, kernel_size=3, stride=1, padding=1,
                                            rp_ratio=rp_ratio, epsilon=rp_eps, bias=False, use_rp=use_rp)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 为了向后兼容，保留BasicBlock的引用
BasicBlock = RandomProjectionBasicBlock