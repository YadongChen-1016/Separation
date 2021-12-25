from torch import nn
import torch
import math
from torch.nn import functional as F
from torch import Tensor
from CrIssCrossAttention import CC_module as CCA


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1) # b, c, h
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1) # b, c
        theta = self.fc1(theta) # b, c/r
        theta = self.relu(theta) # b, c/r
        theta = self.fc2(theta) # b, 2k
        theta = 2 * self.sigmoid(theta) - 1 # b, 2k
        return theta # b, 2k

    def forward(self, x):
        raise NotImplementedError




class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x) # b, 2kc

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            po = relu_coefs[:, :, :self.k]
            # print(po.shape)
            output = x_perm * relu_coefs[:, :, :self.k]
            # print(output.shape)
            output = output + relu_coefs[:, :, self.k:]
            # print(output.shape)
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result



class ACBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(ACBlock,self).__init__()

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel,
                kernel_size=(3, 1), stride=(1, 1), padding=(2 ** dilation, 0), dilation=(2 ** dilation, 1)))
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel,
                kernel_size=(1, 3), stride=(1, 1), padding=(0, 2 ** dilation), dilation=(1, 2 ** dilation)))


    def forward(self, input):

        output2 = self.conv2(input)
        output3 = self.conv3(input)
        output = output2 + output3

        return output


# class ACBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, dilation):
#         super(ACBlock,self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_channel))
#
#         self.conv2 = nn.Sequential(
#             nn.BatchNorm1d(in_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channel, out_channel,
#                 kernel_size=3, stride=1, padding=2 ** dilation, dilation=2 ** dilation),
#             nn.Softmax(-1))
#         self.conv3 = nn.Sequential(
#             nn.BatchNorm1d(in_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channel, out_channel,
#                 kernel_size=3, stride=1, padding=2 ** dilation, dilation=2 ** dilation),
#             nn.Softmax(-1))
#
#
#     def forward(self, input):
#         b, c, f, t = input.size()
#         resdual = self.conv1(input)
#         input_t = F.adaptive_avg_pool2d(input, (1, t)).squeeze(-2)
#         input_f = F.adaptive_avg_pool2d(input, (f, 1)).squeeze(-1)
#
#         output2 = self.conv2(input_t).unsqueeze(-2) * resdual
#         output3 = self.conv3(input_f).unsqueeze(-1) * resdual
#         output = output2 + output3 + resdual
#
#         return output





class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MDenseNet(nn.Module):
    def __init__(self,in_channel,out_channel,first_channel=32,drop_rate = 0.1):
        super(MDenseNet,self).__init__()
        self.model = _MDenseNet(in_channel,first_channel = first_channel,drop_rate = drop_rate)
        self.out = nn.Sequential(
            _DenseBlock(
                2, 16, 32, 0.1),
            nn.Conv2d(32, out_channel, 1)
        )
    def forward(self, input):
        return self.out(self.model(input))


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate, memory_efficient=False, dilation=1):
        super(_DenseLayer, self).__init__()

        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate,
        #                                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
        #




        self.conv_block = nn.Sequential(
            ACBlock(num_input_features, growth_rate, dilation=1),
            # ACBlock(growth_rate, growth_rate, dilation+1),
            nn.Sequential(
                    nn.BatchNorm2d(growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, dilation=1, #padding=2**(dilation+1), dilation=2**(dilation+1),
                        bias=False))
        )

        # self.conv_skip = nn.Sequential(
        #     nn.Conv2d(num_input_features, growth_rate, kernel_size=1),
        #     nn.BatchNorm2d(growth_rate),
        # )

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x


    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        prev_features = input
        # new_features = self.conv1(self.relu1(self.norm1(prev_features)))

        # residual = self.conv_skip(prev_features)
        new_features = self._pad(self.conv_block(prev_features), prev_features) #+ residual


        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features



class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """
    def __init__(self, in_channels):
        super(FReLU,self).__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x= torch.max(x, x1)
        return x


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class eSEModule(nn.Module):
    def __init__(self, channel):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel,channel, kernel_size=1,
                             padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x



class _DenseBlock(nn.Module):
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        self.recurrence = 2
        self.num_input_features = num_input_features
        out_channel = num_input_features + num_layers * growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                dilation=i
            )
            self.layers['denselayer%d' % (i + 1)] = layer
            num_input_features = growth_rate

        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, growth_rate, 1)
        )

        # self.conva = nn.Sequential(
        #     nn.BatchNorm2d(growth_rate),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(growth_rate, growth_rate, 3, padding=1, bias=False))
        # self.cca = CCA(growth_rate, reduction=2)
        # self.convb = nn.Sequential(
        #     nn.BatchNorm2d(growth_rate),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(growth_rate, growth_rate, 3, padding=1, bias=False))
        #
        # self.bottleneck = nn.Sequential(
        #     nn.BatchNorm2d(growth_rate + growth_rate),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(growth_rate + growth_rate, growth_rate, kernel_size=3, padding=1, dilation=1, bias=False))







        self.conv_skip = nn.Sequential(
            nn.Conv2d(self.num_input_features, growth_rate, kernel_size=1),
            nn.BatchNorm2d(growth_rate),
        )

    def forward(self, init_features):
        features_list = [init_features]
        residual = self.conv_skip(init_features)
        for name, layer in self.layers.items():
            init_features = layer(init_features)
            features_list.append(init_features)
        output = self.conv1x1(torch.cat(features_list, dim=1)) + residual


        return output




def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


a = torch.randn(2, 4, 3, 4)
# print(DyReLUB(4)(a).shape)
