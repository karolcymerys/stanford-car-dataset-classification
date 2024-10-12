import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, output_channels, downsample=False, stride=(2, 2), bias=False):
        super(BasicBlock, self).__init__()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, output_channels, kernel_size=(1, 1), stride=stride, bias=bias),
                nn.BatchNorm2d(output_channels)
            )
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=bias),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          bias=bias),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        return F.relu(self.shortcut(x) + self.main(x))


class BasicResnet(nn.Module):
    def __init__(self, layers, in_channels=3, classes=1000, bias=False):
        super(BasicResnet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        self.layer1 = BasicResnet.__build_layer(64, layers[0])
        self.layer2 = BasicResnet.__build_layer(128, layers[1], downsample=True)
        self.layer3 = BasicResnet.__build_layer(256, layers[2], downsample=True)
        self.layer4 = BasicResnet.__build_layer(512, layers[3], downsample=True)
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1, -1),
            nn.Linear(512, classes)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def init_weights(self) -> None:
        self.__init_weights(self.layer0)
        self.__init_weights(self.layer1)
        self.__init_weights(self.layer2)
        self.__init_weights(self.layer3)
        self.__init_weights(self.layer4)
        self.__init_weights(self.layer5)

    @staticmethod
    def __init_weights(module: nn.Module) -> None:
        for layer in module:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    @staticmethod
    def __build_layer(channels, no_layers, downsample=False):
        layers = []
        if downsample:
            layers.append(BasicBlock(int(channels / 2), channels, downsample=True))
        else:
            layers.append(BasicBlock(channels, channels))
        layers.extend([BasicBlock(channels, channels) for _ in range(no_layers - 1)])

        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    def __init__(self, dimmensions, downsample=False, in_channels=None, stride=(2, 2), bias=False):
        super(Bottleneck, self).__init__()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 4 * dimmensions, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(4 * dimmensions)
            )
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, dimmensions, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(dimmensions),
                nn.ReLU(inplace=True),
                nn.Conv2d(dimmensions, dimmensions, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
                nn.BatchNorm2d(dimmensions),
                nn.ReLU(inplace=True),
                nn.Conv2d(dimmensions, 4 * dimmensions, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(4 * dimmensions)
            )
        else:
            self.shortcut = nn.Sequential()
            self.main = nn.Sequential(
                nn.Conv2d(4 * dimmensions, dimmensions, kernel_size=(1, 1), stride=(1, 1), bias=bias),
                nn.BatchNorm2d(dimmensions),
                nn.ReLU(inplace=True),
                nn.Conv2d(dimmensions, dimmensions, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                nn.BatchNorm2d(dimmensions),
                nn.ReLU(inplace=True),
                nn.Conv2d(dimmensions, 4 * dimmensions, kernel_size=(1, 1), stride=(1, 1), bias=bias),
                nn.BatchNorm2d(4 * dimmensions)
            )

    def forward(self, x):
        return F.relu(self.shortcut(x) + self.main(x))


class BottleneckResnet(nn.Module):
    def __init__(self, layers, in_channels=3, classes=1000, bias=False):
        super(BottleneckResnet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        self.layer1 = BottleneckResnet.__build_layer(64, 64, layers[0])
        self.layer2 = BottleneckResnet.__build_layer(256, 128, layers[1])
        self.layer3 = BottleneckResnet.__build_layer(512, 256, layers[2])
        self.layer4 = BottleneckResnet.__build_layer(1024, 512, layers[3])
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1, -1),
            nn.Linear(2048, classes)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def init_weights(self) -> None:
        self.__init_weights(self.layer0)
        self.__init_weights(self.layer1)
        self.__init_weights(self.layer2)
        self.__init_weights(self.layer3)
        self.__init_weights(self.layer4)
        self.__init_weights(self.layer5)

    @staticmethod
    def __init_weights(module: nn.Module) -> None:
        for layer in module:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    @staticmethod
    def __build_layer(in_channels, channels, no_layers):
        layers = [Bottleneck(channels, downsample=True, in_channels=in_channels)]
        layers.extend([Bottleneck(channels) for _ in range(no_layers - 1)])

        return nn.Sequential(*layers)


def resnet18(in_channels=3, classes=1000):
    return BasicResnet([2, 2, 2, 2], in_channels=in_channels, classes=classes)


def resnet34(in_channels=3, classes=1000):
    return BasicResnet([3, 4, 6, 3], in_channels=in_channels, classes=classes)


def resnet50(in_channels=3, classes=1000):
    return BottleneckResnet([3, 4, 6, 3], in_channels=in_channels, classes=classes)


def resnet101(in_channels=3, classes=1000):
    return BottleneckResnet([3, 4, 23, 3], in_channels=in_channels, classes=classes)


def resnet152(in_channels=3, classes=1000):
    return BottleneckResnet([3, 8, 36, 3], in_channels=in_channels, classes=classes)
