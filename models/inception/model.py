from typing import Tuple

import torch
from torch import nn


class InceptionCNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 padding: Tuple[int, int]) -> None:
        super(InceptionCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)

    def init_weights(self) -> None:
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class InceptionModuleV1(nn.Module):
    def __init__(self,
                 in_channels: int,
                 branches_channels: Tuple[int, Tuple[int, int], Tuple[int, int], int]) -> None:
        super(InceptionModuleV1, self).__init__()
        self.branch1 = InceptionCNN(in_channels, branches_channels[0], (1, 1), (1, 1), (0, 0))
        self.branch2 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[1][0], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[1][0], branches_channels[1][1], (3, 3), (1, 1), (1, 1)),
        )
        self.branch3 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[2][0], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[2][0], branches_channels[2][1], (5, 5), (1, 1), (2, 2)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
            InceptionCNN(in_channels, branches_channels[3], (1, 1), (1, 1), (0, 0)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branch1_out = self.branch1(features)
        branch2_out = self.branch2(features)
        branch3_out = self.branch3(features)
        branch4_out = self.branch4(features)
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()


class InceptionAuxClassifier(nn.Module):
    def __init__(self, classes: int) -> None:
        super(InceptionAuxClassifier, self).__init__()
        self.aux = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            InceptionCNN(512, 128, (1, 1), (1, 1), (0, 0)),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, 1024),
            nn.Linear(1024, classes),
            nn.Dropout(p=0.7)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.aux(features)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()


class InceptionV1(nn.Module):
    def __init__(self, classes: int = 1000) -> None:
        super(InceptionV1, self).__init__()

        self.conv1 = InceptionCNN(3, 64, (7, 7), (2, 2), (3, 3))
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.conv2 = InceptionCNN(64, 64, (1, 1), (1, 1), (0, 0))

        self.conv3 = InceptionCNN(64, 192, (3, 3), (1, 1), (1, 1))
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception3a = InceptionModuleV1(192, (64, (96, 128), (16, 32), 32))
        self.inception3b = InceptionModuleV1(256, (128, (128, 192), (32, 96), 64))
        self.maxpool3 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception4a = InceptionModuleV1(480, (192, (96, 208), (16, 48), 64))
        self.inception4b = InceptionModuleV1(512, (160, (112, 224), (24, 64), 64))
        self.inception4c = InceptionModuleV1(512, (128, (128, 256), (24, 64), 64))
        self.inception4d = InceptionModuleV1(512, (112, (144, 288), (32, 64), 64))
        self.inception4e = InceptionModuleV1(528, (256, (160, 320), (32, 128), 128))
        self.maxpool4 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True)

        self.inception5a = InceptionModuleV1(832, (256, (160, 320), (32, 128), 128))
        self.inception5b = InceptionModuleV1(832, (384, (192, 384), (48, 128), 128))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
            nn.Linear(1024, classes)
        )
        self.classifier_aux1 = InceptionAuxClassifier(classes)
        self.classifier_aux2 = InceptionAuxClassifier(classes)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv1_out = self.conv1(features)
        maxpool1_out = self.maxpool1(conv1_out)
        conv2_out = self.conv2(maxpool1_out)
        conv3_out = self.conv3(conv2_out)
        maxpool2_out = self.maxpool2(conv3_out)

        inception3a_out = self.inception3a(maxpool2_out)
        inception3b_out = self.inception3b(inception3a_out)
        maxpool3_out = self.maxpool3(inception3b_out)

        inception4a_out = self.inception4a(maxpool3_out)
        inception4b_out = self.inception4b(inception4a_out)
        inception4c_out = self.inception4c(inception4b_out)
        inception4d_out = self.inception4d(inception4c_out)
        inception4e_out = self.inception4e(inception4d_out)
        maxpool4_out = self.maxpool4(inception4e_out)

        inception5a_out = self.inception5a(maxpool4_out)
        inception5b_out = self.inception5b(inception5a_out)

        clasifier_out = self.classifier(inception5b_out)
        classifier_aux_1 = self.classifier_aux1(inception4a_out)
        classifier_aux_2 = self.classifier_aux1(inception4c_out)

        return (
            clasifier_out,
            classifier_aux_1,
            classifier_aux_2
        )

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()
            elif isinstance(layer, InceptionModuleV1):
                layer.init_weights()
            elif isinstance(layer, InceptionAuxClassifier):
                layer.init_weights()
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()


class InceptionModuleV3A(nn.Module):
    def __init__(self,
                 in_channels: int,
                 branches_channels: Tuple[int, Tuple[int, int], Tuple[int, int], int]) -> None:
        super(InceptionModuleV3A, self).__init__()
        self.branch1 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[0], (1, 1), (1, 1), (0, 0))
        )
        self.branch2 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[1][0], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[1][0], branches_channels[1][1], (3, 3), (1, 1), (1, 1)),
            InceptionCNN(branches_channels[1][1], branches_channels[1][1], (3, 3), (1, 1), (1, 1)),
        )
        self.branch3 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[2][0], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[2][0], branches_channels[2][1], (5, 5), (1, 1), (2, 2)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
            InceptionCNN(in_channels, branches_channels[3], (1, 1), (1, 1), (0, 0)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branch1_out = self.branch1(features)
        branch2_out = self.branch2(features)
        branch3_out = self.branch3(features)
        branch4_out = self.branch4(features)
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()


class InceptionModuleV3B(nn.Module):
    def __init__(self,
                 in_channels: int,
                 branches_channels: Tuple[int, Tuple[int, int]]) -> None:
        super(InceptionModuleV3B, self).__init__()
        self.branch1 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[0], (3, 3), (2, 2), (0, 0))
        )
        self.branch2 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[1][0], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[1][0], branches_channels[1][1], (3, 3), (1, 1), (1, 1)),
            InceptionCNN(branches_channels[1][1], branches_channels[1][1], (3, 3), (2, 2), (0, 0))
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d((3, 3), (2, 2))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branch1_out = self.branch1(features)
        branch2_out = self.branch2(features)
        branch3_out = self.branch3(features)
        return torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()


class InceptionModuleV3C(nn.Module):
    def __init__(self,
                 in_channels: int,
                 branches_channels: Tuple[int, Tuple[int, int], Tuple[int, int], int]) -> None:
        super(InceptionModuleV3C, self).__init__()
        self.branch1 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[0], (1, 1), (1, 1), (0, 0))
        )
        self.branch2 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[1][0], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[1][0], branches_channels[1][0], (7, 1), (1, 1), (3, 0)),
            InceptionCNN(branches_channels[1][0], branches_channels[1][0], (1, 7), (1, 1), (0, 3)),
            InceptionCNN(branches_channels[1][0], branches_channels[1][0], (7, 1), (1, 1), (3, 0)),
            InceptionCNN(branches_channels[1][0], branches_channels[1][1], (1, 7), (1, 1), (0, 3)),
        )
        self.branch3 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[2][0], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[2][0], branches_channels[2][0], (7, 1), (1, 1), (3, 0)),
            InceptionCNN(branches_channels[2][0], branches_channels[2][1], (1, 7), (1, 1), (0, 3)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
            InceptionCNN(in_channels, branches_channels[3], (1, 1), (1, 1), (0, 0)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branch1_out = self.branch1(features)
        branch2_out = self.branch2(features)
        branch3_out = self.branch3(features)
        branch4_out = self.branch4(features)
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()


class InceptionModuleV3D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 branches_channels: Tuple[Tuple[int, int], int]) -> None:
        super(InceptionModuleV3D, self).__init__()
        self.branch1 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[0][1], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[0][1], branches_channels[0][1], (3, 3), (2, 2), (0, 0))
        )
        self.branch2 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[1], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[1], branches_channels[1], (1, 7), (1, 1), (0, 3)),
            InceptionCNN(branches_channels[1], branches_channels[1], (7, 1), (1, 1), (3, 0)),
            InceptionCNN(branches_channels[1], branches_channels[1], (3, 3), (2, 2), (0, 0))
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d((3, 3), (2, 2))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branch1_out = self.branch1(features)
        branch2_out = self.branch2(features)
        branch3_out = self.branch3(features)
        return torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()


class InceptionModuleV3E(nn.Module):
    def __init__(self,
                 in_channels: int,
                 branches_channels: Tuple[int, Tuple[int, int], int, int]) -> None:
        super(InceptionModuleV3E, self).__init__()
        self.branch1 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[0], (1, 1), (1, 1), (0, 0))
        )
        self.branch2 = nn.Sequential(
            InceptionCNN(in_channels, branches_channels[1][0], (1, 1), (1, 1), (0, 0)),
            InceptionCNN(branches_channels[1][0], branches_channels[1][1], (3, 3), (1, 1), (1, 1)),
        )
        self.branch2a = InceptionCNN(branches_channels[1][1], branches_channels[1][1], (1, 3), (1, 1), (0, 1))
        self.branch2b = InceptionCNN(branches_channels[1][1], branches_channels[1][1], (3, 1), (1, 1), (1, 0))
        self.branch3 = InceptionCNN(in_channels, branches_channels[2], (1, 1), (1, 1), (0, 0))
        self.branch3a = InceptionCNN(branches_channels[2], branches_channels[2], (1, 3), (1, 1), (0, 1))
        self.branch3b = InceptionCNN(branches_channels[2], branches_channels[2], (3, 1), (1, 1), (1, 0))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
            InceptionCNN(in_channels, branches_channels[3], (1, 1), (1, 1), (0, 0)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branch1_out = self.branch1(features)
        branch2_out = self.branch2(features)
        branch2a_out = self.branch2a(branch2_out)
        branch2b_out = self.branch2b(branch2_out)
        branch3_out = self.branch3(features)
        branch3a_out = self.branch3a(branch3_out)
        branch3b_out = self.branch3b(branch3_out)
        branch4_out = self.branch4(features)
        return torch.cat([branch1_out, branch2a_out, branch2b_out, branch3a_out, branch3b_out, branch4_out], dim=1)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()


class InceptionV3(nn.Module):
    def __init__(self, classes: int = 1000) -> None:
        super(InceptionV3, self).__init__()

        self.conv1 = InceptionCNN(3, 32, (3, 3), (2, 2), (0, 0))
        self.conv2 = InceptionCNN(32, 32, (3, 3), (1, 1), (0, 0))
        self.conv3 = InceptionCNN(32, 64, (3, 3), (1, 1), (1, 1))
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.conv4 = InceptionCNN(64, 80, (1, 1), (1, 1), (0, 0))
        self.conv5 = InceptionCNN(80, 192, (3, 3), (1, 1), (0, 0))
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception6a = InceptionModuleV3A(192, (64, (64, 96), (48, 64), 32))
        self.inception6b = InceptionModuleV3A(256, (64, (64, 96), (48, 64), 64))
        self.inception6c = InceptionModuleV3A(288, (64, (64, 96), (48, 64), 64))
        self.maxpool3 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception7a = InceptionModuleV3B(288, (384, (64, 96)))
        self.inception7b = InceptionModuleV3C(768, (192, (128, 192), (128, 192), 192))
        self.inception7c = InceptionModuleV3C(768, (192, (160, 192), (160, 192), 192))
        self.inception7d = InceptionModuleV3C(768, (192, (160, 192), (160, 192), 192))
        self.inception7e = InceptionModuleV3C(768, (192, (192, 192), (192, 192), 192))

        self.inception8a = InceptionModuleV3D(768, ((192, 320), 192))
        self.inception8b = InceptionModuleV3E(1280, (320, (448, 384), 384, 192))
        self.inception8c = InceptionModuleV3E(2048, (320, (448, 384), 384, 192))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
            nn.Linear(2048, classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        conv1_out = self.conv1(features)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        maxpool1_out = self.maxpool1(conv3_out)

        conv4_out = self.conv4(maxpool1_out)
        conv5_out = self.conv5(conv4_out)
        maxpool2_out = self.maxpool2(conv5_out)

        inception6a_out = self.inception6a(maxpool2_out)
        inception6b_out = self.inception6b(inception6a_out)
        inception6c_out = self.inception6c(inception6b_out)

        inception7a_out = self.inception7a(inception6c_out)
        inception7b_out = self.inception7b(inception7a_out)
        inception7c_out = self.inception7c(inception7b_out)
        inception7d_out = self.inception7d(inception7c_out)
        inception7e_out = self.inception7e(inception7d_out)

        inception8a_out = self.inception8a(inception7e_out)
        inception8b_out = self.inception8b(inception8a_out)
        inception8c_out = self.inception8c(inception8b_out)

        return self.classifier(inception8c_out)

    def init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, InceptionCNN):
                layer.init_weights()
            elif isinstance(layer, InceptionModuleV1):
                layer.init_weights()
            elif isinstance(layer, InceptionAuxClassifier):
                layer.init_weights()
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
