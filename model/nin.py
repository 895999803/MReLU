import torch.nn as nn
from activation import Activate


class ConvReLu(nn.Module):

    def __init__(self, conv, out_channel, activate, init=1.0):
        super(ConvReLu, self).__init__()

        self.conv = conv
        self.bn = nn.BatchNorm2d(out_channel)
        self.activate = Activate(activate, init=init)

        self.act_value = activate

    def forward(self, x_input):
        x = self.conv(x_input)
        x = self.activate(x)
        return x


class Model(nn.Module):

    def __init__(self, num_classes, activate=1, init=1.0):
        super(Model, self).__init__()

        self.num_classes = num_classes

        self.conv1 = ConvReLu(nn.Conv2d(3, 192, 5, padding=2, bias=False), 192, activate, init=init)
        self.conv2 = ConvReLu(nn.Conv2d(192, 160, 1), 160, activate, init=init)
        self.conv3 = ConvReLu(nn.Conv2d(160, 96, 1), 96, activate, init=init)
        self.maxpooling = nn.MaxPool2d(3, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv4 = ConvReLu(nn.Conv2d(96, 192, 5, padding=2, bias=False), 192, activate, init=init)
        self.conv5 = ConvReLu(nn.Conv2d(192, 192, 1), 192, activate, init=init)
        self.conv6 = ConvReLu(nn.Conv2d(192, 192, 1), 192, activate, init=init)
        self.avgpooling = nn.AvgPool2d(3, stride=2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.conv7 = ConvReLu(nn.Conv2d(192, 192, 3, padding=1, bias=False), 192, activate, init=init)
        self.conv8 = ConvReLu(nn.Conv2d(192, 192, 1), 192, activate, init=init)
        self.conv9 = nn.Conv2d(192, self.num_classes, 1)
        self.globalpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_input):
        x_input = self.conv1(x_input)
        x_input = self.conv2(x_input)
        x_input = self.conv3(x_input, is_show=True)

        x_input = self.maxpooling(x_input)
        x_input = self.dropout1(x_input)

        x_input = self.conv4(x_input)
        x_input = self.conv5(x_input)
        x_input = self.conv6(x_input)

        x_input = self.avgpooling(x_input)
        x_input = self.dropout2(x_input)

        x_input = self.conv7(x_input)
        x_input = self.conv8(x_input)
        x_input = self.conv9(x_input)

        x_input = self.globalpooling(x_input)
        x_input = x_input.view(x_input.size(0), self.num_classes)

        return x_input



