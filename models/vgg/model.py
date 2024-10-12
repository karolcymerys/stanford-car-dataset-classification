from typing import Tuple

import torch
from torch import nn


class VGGBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1)) -> None:
        super(VGGBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.block(features)

    def init_weights(self) -> None:
        for layer in self.block:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class VGGLayer(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 blocks: int) -> None:
        super(VGGLayer, self).__init__()

        layer_blocks = [VGGBlock(input_channels, output_channels)]
        for _ in range(blocks - 1):
            layer_blocks.append(VGGBlock(output_channels, output_channels))
        layer_blocks.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.blocks = nn.Sequential(*layer_blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.blocks(features)

    def init_weights(self) -> None:
        for layer in self.blocks:
            if isinstance(layer, VGGBlock):
                layer.init_weights()


class VGGClassifier(nn.Module):
    def __init__(self,
                 classes: int = 1000) -> None:
        super(VGGClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def init_weights(self) -> None:
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()


class VGG11(nn.Module):
    def __init__(self,
                 classes: int = 1000) -> None:
        super(VGG11, self).__init__()

        self.model = nn.Sequential(
            VGGLayer(3, 64, 1),
            VGGLayer(64, 128, 1),
            VGGLayer(128, 256, 2),
            VGGLayer(256, 512, 2),
            VGGLayer(512, 512, 2),
            nn.AdaptiveAvgPool2d((7, 7)),
            VGGClassifier(classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def init_weights(self) -> None:
        for layer in self.model:
            if isinstance(layer, VGGLayer):
                layer.init_weights()
            elif isinstance(layer, VGGClassifier):
                layer.init_weights()


class VGG13(nn.Module):
    def __init__(self,
                 classes: int = 1000) -> None:
        super(VGG13, self).__init__()

        self.model = nn.Sequential(
            VGGLayer(3, 64, 2),
            VGGLayer(64, 128, 2),
            VGGLayer(128, 256, 2),
            VGGLayer(256, 512, 2),
            VGGLayer(512, 512, 2),
            nn.AdaptiveAvgPool2d((7, 7)),
            VGGClassifier(classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def init_weights(self) -> None:
        for layer in self.model:
            if isinstance(layer, VGGLayer):
                layer.init_weights()
            elif isinstance(layer, VGGClassifier):
                layer.init_weights()


class VGG16(nn.Module):
    def __init__(self,
                 classes: int = 1000) -> None:
        super(VGG16, self).__init__()

        self.model = nn.Sequential(
            VGGLayer(3, 64, 2),
            VGGLayer(64, 128, 2),
            VGGLayer(128, 256, 3),
            VGGLayer(256, 512, 3),
            VGGLayer(512, 512, 3),
            nn.AdaptiveAvgPool2d((7, 7)),
            VGGClassifier(classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def init_weights(self) -> None:
        for layer in self.model:
            if isinstance(layer, VGGLayer):
                layer.init_weights()
            elif isinstance(layer, VGGClassifier):
                layer.init_weights()


class VGG19(nn.Module):
    def __init__(self,
                 classes: int = 1000) -> None:
        super(VGG19, self).__init__()

        self.model = nn.Sequential(
            VGGLayer(3, 64, 2),
            VGGLayer(64, 128, 2),
            VGGLayer(128, 256, 4),
            VGGLayer(256, 512, 4),
            VGGLayer(512, 512, 4),
            nn.AdaptiveAvgPool2d((7, 7)),
            VGGClassifier(classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def init_weights(self) -> None:
        for layer in self.model:
            if isinstance(layer, VGGLayer):
                layer.init_weights()
            elif isinstance(layer, VGGClassifier):
                layer.init_weights()
