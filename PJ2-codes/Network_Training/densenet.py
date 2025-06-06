import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bottleneck block for DenseNet
class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


# Transition layer
class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


# Base DenseNet model
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# DenseNet-121 definition
def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


# DenseNet-BC-100 definition (Basic Compression version)
class DenseNetBC100(nn.Module):
    def __init__(self, growth_rate=12, num_classes=10):
        super(DenseNetBC100, self).__init__()
        self.growth_rate = growth_rate
        num_init_features = 2 * growth_rate  # =24
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=3, padding=1, bias=False)

        # Block 1
        self.block1 = self._make_dense_block(num_layers=16, in_channels=num_init_features)
        num_channels = num_init_features + 16 * growth_rate
        self.trans1 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        # Block 2
        self.block2 = self._make_dense_block(num_layers=16, in_channels=num_channels)
        num_channels = num_channels + 16 * growth_rate
        self.trans2 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        # Block 3
        self.block3 = self._make_dense_block(num_layers=16, in_channels=num_channels)
        num_channels = num_channels + 16 * growth_rate

        # Final layers
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, num_layers, in_channels):
        layers = []
        for _ in range(num_layers):
            layers.append(Bottleneck(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# MyDenseNetBC100 with SEBlock and dropout
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, use_se=True, drop_prob=0.0):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.se = SEBlock(growth_rate) if use_se else nn.Identity()
        self.drop_prob = drop_prob

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.se(out)
        if self.drop_prob > 0:
            out = F.dropout(out, p=self.drop_prob, training=self.training)
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return self.pool(out)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, **kwargs):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Bottleneck(in_channels, growth_rate, **kwargs))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MyDenseNetBC100(nn.Module):
    def __init__(self, growth_rate=12, reduction=0.5, num_classes=10, drop_prob=0.1):
        super(MyDenseNetBC100, self).__init__()
        num_blocks = [16, 16, 16]  # DenseNet-BC-100
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.block1 = DenseBlock(num_blocks[0], num_channels, growth_rate, drop_prob=drop_prob)
        num_channels += num_blocks[0] * growth_rate
        self.trans1 = Transition(num_channels, int(num_channels * reduction))
        num_channels = int(num_channels * reduction)
        self.block2 = DenseBlock(num_blocks[1], num_channels, growth_rate, drop_prob=drop_prob)
        num_channels += num_blocks[1] * growth_rate
        self.trans2 = Transition(num_channels, int(num_channels * reduction))
        num_channels = int(num_channels * reduction)
        self.block3 = DenseBlock(num_blocks[2], num_channels, growth_rate, drop_prob=drop_prob)
        num_channels += num_blocks[2] * growth_rate
        self.bn = nn.BatchNorm2d(num_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = self.pool(out).view(out.size(0), -1)
        return self.fc(out)


# Mish version of MyDenseNetBC100 with Mish activation
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_prob=0.0):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.se = SEBlock(growth_rate)
        self.drop_prob = drop_prob
        self.act = Mish()

    def forward(self, x):
        out = self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        out = self.se(out)
        if self.drop_prob > 0:
            out = F.dropout(out, p=self.drop_prob, training=self.training)
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)
        self.act = Mish()

    def forward(self, x):
        out = self.act(self.bn(x))
        out = self.conv(out)
        return self.pool(out)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_prob=0.0):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Bottleneck(in_channels, growth_rate, drop_prob=drop_prob))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MyDenseNetBC100_Mish(nn.Module):
    def __init__(self, growth_rate=12, reduction=0.5, num_classes=10, drop_prob=0.1):
        super(MyDenseNetBC100_Mish, self).__init__()
        num_blocks = [16, 16, 16]
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.block1 = DenseBlock(num_blocks[0], num_channels, growth_rate, drop_prob=drop_prob)
        num_channels += num_blocks[0] * growth_rate
        self.trans1 = Transition(num_channels, int(num_channels * reduction))
        num_channels = int(num_channels * reduction)
        self.block2 = DenseBlock(num_blocks[1], num_channels, growth_rate, drop_prob=drop_prob)
        num_channels += num_blocks[1] * growth_rate
        self.trans2 = Transition(num_channels, int(num_channels * reduction))
        num_channels = int(num_channels * reduction)
        self.block3 = DenseBlock(num_blocks[2], num_channels, growth_rate, drop_prob=drop_prob)
        num_channels += num_blocks[2] * growth_rate
        self.bn = nn.BatchNorm2d(num_channels)
        self.se_final = SEBlock(num_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)
        self.act = Mish()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.act(self.bn(out))
        out = self.se_final(out)
        out = self.pool(out).view(out.size(0), -1)
        return self.fc(out)