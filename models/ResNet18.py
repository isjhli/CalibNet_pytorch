import torch.nn as nn
from torch.nn import functional as F
import torch


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, inplanes=3, planes=64):
        super(ResNet18, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, planes, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(planes, planes, stride=1, dilation=1),
            BasicBlock(planes, planes, stride=1)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(planes, 2 * planes, stride=2, dilation=1, downsample=nn.Sequential(
                nn.Conv2d(planes, 2 * planes, 1, stride=2, dilation=1, bias=False),
                nn.BatchNorm2d(2 * planes)
            )),
            BasicBlock(2 * planes, 2 * planes, stride=1)
        )

        # TODO layer3 和 layer4 没有按照论文中对宽和高进行减半
        self.layer3 = nn.Sequential(
            BasicBlock(2 * planes, 4 * planes, stride=1, dilation=2, downsample=nn.Sequential(
                nn.Conv2d(2 * planes, 4 * planes, 1, stride=1, bias=False),
                nn.BatchNorm2d(4 * planes)
            )),
            BasicBlock(4 * planes, 4 * planes, stride=1, dilation=2)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(4 * planes, 8 * planes, stride=1, dilation=2, downsample=nn.Sequential(
                nn.Conv2d(4 * planes, 8 * planes, 1, stride=1, bias=False),
                nn.BatchNorm2d(8 * planes)
            )),
            BasicBlock(8 * planes, 8 * planes, stride=1, dilation=4)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4


if __name__ == '__main__':
    model = ResNet18()
    x = torch.randn(1, 3, 1242, 375)
    outs = model(x)
    print(outs[0].size(), outs[1].size(), outs[2].size(), outs[3].size())
