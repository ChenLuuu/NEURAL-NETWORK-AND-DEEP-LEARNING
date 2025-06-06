import torch
import torch.nn as nn
import torch.nn.functional as F

# This is a simple test
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
        

class CustomCNN(nn.Module):
    def __init__(self, activation_fn=F.relu):
        super(CustomCNN, self).__init__()
        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 2D Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # Activation function
        self.activation_fn = activation_fn
        
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.activation_fn(x)  # Using the specified activation
        x = self.pool(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.pool(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.pool(x)
        
        # Flatten the output
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn=F.relu):
        super(ResidualBlock, self).__init__()
        self.activation_fn = activation_fn
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut layer (use 1x1 conv if in_channels != out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation_fn(out)
        return out


class ModernCNN(nn.Module):
    def __init__(self, activation_fn=F.relu, dropout_rate=0.5):
        super(ModernCNN, self).__init__()
        self.activation_fn = activation_fn
        self.pool = nn.MaxPool2d(2, 2)

        # First conv block (no residual needed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual blocks
        self.resblock1 = ResidualBlock(32, 64, activation_fn)
        self.resblock2 = ResidualBlock(64, 128, activation_fn)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.pool(x)

        # Residual block 1
        x = self.resblock1(x)
        x = self.pool(x)

        # Residual block 2
        x = self.resblock2(x)
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x