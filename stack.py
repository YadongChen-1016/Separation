import torch
import math
import torch.nn as nn
from torch.autograd import Function
from _mdensenet import _DenseBlock
from torch.nn import functional as F
# from TFAttention import TFA
from CrIssCrossAttention import CC_module as CCA




def judge(input, output, predict):
    return torch.mean(torch.abs(predict.mul(input) - output))




class MMDenseNet(nn.Module):
    def __init__(self, input_channel, drop_rate=0.1):
        super(MMDenseNet, self).__init__()
        kl_low = [(14,4), (16,4), (16,4), (16,4), (16,4), (16,4), (16, 4)]
        kl_high = [(10,3), (10,3), (10,3), (10,3), (10,3), (10,3), (16, 3)]
        kl_full = [(6, 2), (6, 2), (6, 2), (6, 4), (6, 2), (6, 2), (6, 2)]
        in_size = [31, 32, 63]
        hidden = [128, 32, 128]
        self.lowNet = _MDenseNet_STEM(input_channel=input_channel, first_channel=32, first_kernel=(4, 3), scale=3, kl=kl_low, drop_rate=drop_rate, hidden=hidden[0], in_size=in_size[0])
        self.highNet = _MDenseNet_STEM(input_channel=input_channel, first_channel=32, first_kernel=(3, 3), scale=3, kl=kl_high, drop_rate=drop_rate, hidden=hidden[0], in_size=in_size[1])
        self.fullNet = _MDenseNet_STEM(input_channel=input_channel, first_channel=32, first_kernel=(4, 3), scale=3, kl=kl_full, drop_rate=drop_rate, hidden=hidden[0], in_size=in_size[2])
        # self.low_out = TFA(kl_low[-1][0], kl_low[-1][0])
        # self.high_out = TFA(kl_high[-1][0], kl_high[-1][0])

        last_channel = kl_low[-1][0] + kl_full[-1][0]
        self.out = nn.Sequential(
            _DenseBlock(
                2, last_channel, 32, drop_rate),
            Conv(32, 32, 1))
            # nn.Conv2d(32, 2, kernel_size=(2, 1)),
            # nn.ReLU())

    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x

    def forward(self, input):
        B, C, F, T = input.shape
        # low_input = input[:, :, :F // 2, :]
        # high_input = input[:, :, F // 2:, :]
        low_input = input[:, :, :F // 2, :]
        high_input = input[:, :, F // 2:, :]
        low = self.lowNet(low_input)
        # low = self.low_out(low)
        high = self.highNet(high_input)
        # high = self.high_out(high)
        output = torch.cat([low, high], 2)

        full_output = self.fullNet(input)

        output = torch.cat([output, full_output], 1)
        output = self.out(output)
        output = self._pad(output, input)
        return output


class attention_Module(nn.Module):
    """ self attention module"""

    def __init__(self, in_dim, f=None):
        super(attention_Module, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=4, kernel_size=1)
        self.query_linear = nn.Linear(4*f, 10)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=4, kernel_size=1)
        self.key_linear = nn.Linear(4*f, 10)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)


        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X F X T)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        b, c, f, t = x.size()
        query = self.query_conv(x)    # b x 5 x f x t
        query = query.reshape(query.size(0), query.size(1)*query.size(2), -1) # b x 5*f x t
        query = self.query_linear(query.transpose(2, 1)) # b x t x 20

        key = self.key_conv(x) # b x 5 x f x t
        key = key.view(key.size(0), key.size(1) * key.size(2), -1) # b x 5*f x t
        key = self.key_linear(key.transpose(2, 1)).transpose(2, 1) # b x 20 x t

        energy = torch.matmul(query, key) # b x t x t
        attention = self.softmax(energy) # b x t x t
        value = self.value_conv(x) # b x c x f x t
        value = value.view(value.size(0), value.size(1)*value.size(2), -1) # b x c*f x t

        out = torch.matmul(value, attention.transpose(2, 1)) # b x c*f x t
        out = out.reshape(b, c, f, t) # b x c x f x t

        return out



class LSTMBlock(nn.Module):

    """LSTMBlock."""

    def __init__(self, in_channels, in_size, batch_size, hidden_size=512):
        """Initialize LSTMBlock."""
        self.in_channels = in_channels
        self.in_size = in_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = None
        super(LSTMBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.lstm = nn.LSTM(
            input_size=self.in_size, hidden_size=self.hidden_size,
            bidirectional=True)
        # self.init_hidden(self.batch_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.in_size)

    def init_hidden(self, batch_size):
        """Initialize the hidden state of the RNN."""
        hidden1 = torch.zeros(2, batch_size, self.hidden_size)
        hidden2 = torch.zeros(2, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden1 = hidden1.cuda()
            hidden2 = hidden2.cuda()
        self.hidden = (hidden1, hidden2)


    def forward(self, x):
        """Forward Pass."""
        x = self.conv(x).squeeze(1)
        # batch_size x freq x time
        x, _ = self.lstm(x.permute(2, 0, 1), self.hidden)
        # time x batch_size x hidden_size
        x = self.linear(x.permute(1, 0, 2))
        # batch_size x time x freq
        return x.permute(0, 2, 1).unsqueeze(1)



class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()



class _MDenseNet_STEM(nn.Module):
    def __init__(self, input_channel=2,
                 first_channel=32,
                 first_kernel=(3, 3),
                 scale=3,
                 kl=[(14, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)],
                 drop_rate=0.1,
                 hidden=None,
                 in_size=None):
        super(_MDenseNet_STEM, self).__init__()
        self.first_channel = 32
        self.dual_rnn = []

        # self.first_kernel = first_kernel
        # self.scale = scale
        # self.first_conv = nn.Conv2d(input_channel, first_channel, first_kernel)
        # encoder part
        self.En1 = _DenseBlock(kl[0][1], self.first_channel, kl[0][0], drop_rate)
        self.pool1 = nn.Sequential(
            nn.Conv2d(kl[0][0], kl[0][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.En2 = _DenseBlock(kl[1][1], kl[0][0], kl[1][0], drop_rate)
        self.pool2 = nn.Sequential(
            nn.Conv2d(kl[1][0], kl[1][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.En3 = _DenseBlock(kl[2][1], kl[1][0], kl[2][0], drop_rate)
        self.pool3 = nn.Sequential(
            nn.Conv2d(kl[2][0], kl[2][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # enter part
        self.Enter = _DenseBlock(kl[3][1], kl[2][0], kl[3][0], drop_rate)

        # decoder part
        self.up3 = nn.ConvTranspose2d(kl[3][0], kl[3][0], kernel_size=2, stride=2)
        self.De3 = _DenseBlock(kl[-3][1], kl[3][0]+kl[2][0], kl[-3][0], drop_rate)

        self.up2 = nn.ConvTranspose2d(kl[-3][0], kl[-3][0], kernel_size=2, stride=2)
        self.De2 = _DenseBlock(kl[-2][1], kl[-3][0]+kl[1][0], kl[-2][0], drop_rate)

        self.up1 = nn.ConvTranspose2d(kl[-2][0], kl[-2][0], kernel_size=2, stride=2)
        self.De1 = _DenseBlock(kl[-1][1], kl[-2][0]+kl[0][0], kl[-1][0], drop_rate)


    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x


    def forward(self, input):
        x0 = input
        # x0 = self.first_conv(x0)
        # encoder part
        x1 = self.En1(x0)
        x_1 = self.pool1(x1)
        x2 = self.En2(x_1)
        # x2 = self.E_att2(x2)
        x_2 = self.pool2(x2)
        x3 = self.En3(x_2)
        # x3 = self.E_att3(x3)
        x_3 = self.pool3(x3)

        xy_ = self.Enter(x_3)
        # lstm = self.lstm(xy_)
        # xy = torch.cat([xy_, lstm], dim=1)
        # for i in range(3):
        #     xy_ = self.dual_rnn[i](xy_)
        # xy_ = self.nonlinear(xy_)

        # decoder part
        y3 = self.up3(xy_)
        y_3 = self.De3(torch.cat([self._pad(y3, x3), x3], dim=1))
        # y_3 = self.D_att3(y_3)
        y2 = self.up2(y_3)
        y_2 = self.De2(torch.cat([self._pad(y2, x2), x2], dim=1))
        # y_2 = self.D_att2(y_2)
        y1 = self.up1(y_2)
        y_1 = self.De1(torch.cat([self._pad(y1, x1), x1], dim=1))
        # y_1 = self.D_att1(y_1)

        output = self._pad(y_1, input)
        return output



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, do_activation=True):
        super(Conv, self).__init__()
        if not do_activation:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2))
        else:
            self.model = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2))

    def forward(self, x):
        x = self.model(x)
        return x



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


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 2
        self.conva = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False))
        self.cca = CCA(inter_channels, 4)
        self.convb = nn.Sequential(
            nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False))

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels+inter_channels), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output



class StackedMMDenseNet(nn.Module):
    def __init__(self, num_stacks):
        super(StackedMMDenseNet, self).__init__()
        self.num_stacks = num_stacks
        # self.first_conv = Conv(1, 64, kernel_size=3, stride=1)
        # self.glu0 = GLU(64)
        # self.glu1 = GLU(64)
        # self.glu2 = GLU(64)
        # self.end_conv = Conv(64, 32, kernel_size=3, stride=1)
        self.prepare = nn.Sequential(Conv(1, 64, kernel_size=3, stride=1),
                                     _DenseBlock(3, 64, 32, drop_rate=0.1)
                                     )
                                     # TODO 这里PosNet是stride为3，padding为2，之后可以试试，先保持和SHNet一样



                                     # Conv(64, 128),
                                     # Conv(128, 128),
                                     # Conv(128, 32))

        self.att1 = RCCAModule(32, 32)
        self.att2 = SELayer(32, reduction=4)
        self.stack = nn.ModuleList(
            nn.Sequential(
                MMDenseNet(1)) for i in range(num_stacks))
        self.output = nn.ModuleList(Conv(32, 2) for i in range(num_stacks))
        self.next = nn.ModuleList(nn.Sequential(Conv(32, 32)) for i in range(num_stacks - 1))
        self.merge = nn.ModuleList(Conv(2, 32) for i in range(num_stacks - 1))

        # init
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_normal_(p)


    def forward(self, x):

        x = self.prepare(x)
        x1 = self.att1(x)
        x2 = self.att2(x)
        x = x1 + x2
        # resdual = x
        # 要输出多个堆叠网络的结果
        predicts = []
        for i in range(self.num_stacks):
            x = self.stack[i](x)
            predict = self.output[i](x)
            predicts.append(predict)

            if i != self.num_stacks - 1:
                x = self.next[i](x) + self.merge[i](predicts[-1]) + x

        # 返回的第一维是多个loss，第二维是batch，后面两个通道
        return torch.stack(predicts, 0)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# # a = torch.randn(1,16,300,100)
# model = MMDenseNet(input_channel=16)
# # print(model)
# # print(model(a).size())
# print(get_parameter_number(model))

if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    x = torch.randn(4, 1, 512, 64)
    net = StackedMMDenseNet(4)
    # net = MMDenseNet(1)
    # print(get_parameter_number(net))
    # print(net)
    # y = net(x)
    # print(y.shape)
    macs, params = profile(net, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)