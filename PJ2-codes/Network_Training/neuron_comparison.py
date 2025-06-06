import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
import torch.nn.functional as F

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
print("\nLoading datasets...")
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# Split training set into train and validation
train_size = int(0.99 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])
print(f"Dataset sizes - Train: {len(trainset)}, Validation: {len(valset)}, Test: {len(testset)}")

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

class CustomCNNWithFilters(nn.Module):
    def __init__(self, filters_config):
        super(CustomCNNWithFilters, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, filters_config[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters_config[0])
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(filters_config[0], filters_config[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters_config[1])
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(filters_config[1], filters_config[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters_config[2])
        
        # Fully connected layers
        self.fc1 = nn.Linear(filters_config[2] * 4 * 4, filters_config[3])
        self.fc2 = nn.Linear(filters_config[3], 10)
        
        # Dropout layers
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def evaluate_metrics(model, dataloader, criterion):
    """
    Evaluate model accuracy and loss on the provided dataset
    """
    correct = 0
    total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, total_loss / len(dataloader)

def train_model(model, optimizer, criterion, epochs=10, name="Model"):
    """
    Train the model and return metrics for plotting
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\nTraining {name}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, data in enumerate(pbar, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(i+1):.3f}',
                'acc': f'{100*correct/total:.1f}%'
            })
        
        # Evaluate metrics
        train_acc, train_loss = evaluate_metrics(model, trainloader, criterion)
        val_acc, val_loss = evaluate_metrics(model, valloader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
    
    return train_losses, train_accs, val_losses, val_accs

# Define different model configurations
# Format: [conv1_filters, conv2_filters, conv3_filters, fc1_neurons]
model_configs = {
    'Tiny': [16, 32, 64, 128],
    'Small': [32, 64, 128, 256],
    'Medium': [64, 128, 256, 512],
    'Large': [128, 256, 512, 1024],
    'Extra Large': [256, 512, 1024, 2048]
}

# Train models with different configurations
results = {}
epochs = 10

for name, config in model_configs.items():
    print(f"\nTraining model with {name} configuration...")
    print(f"Filters/Neurons: {config}")
    
    model = CustomCNNWithFilters(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    metrics = train_model(model, optimizer, criterion, epochs=epochs, name=name)
    results[name] = metrics

# Calculate number of parameters for each model
param_counts = {}
for name, config in model_configs.items():
    model = CustomCNNWithFilters(config)
    param_counts[name] = sum(p.numel() for p in model.parameters())
    print(f"\n{name} model parameters: {param_counts[name]:,}")

# Create plots
plt.figure(figsize=(20, 15))

# Training Loss
plt.subplot(3, 2, 1)
for name, (train_losses, _, _, _) in results.items():
    plt.plot(train_losses, label=f"{name} ({param_counts[name]:,} params)")
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Training Accuracy
plt.subplot(3, 2, 2)
for name, (_, train_accs, _, _) in results.items():
    plt.plot(train_accs, label=f"{name} ({param_counts[name]:,} params)")
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Validation Loss
plt.subplot(3, 2, 3)
for name, (_, _, val_losses, _) in results.items():
    plt.plot(val_losses, label=f"{name} ({param_counts[name]:,} params)")
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Validation Accuracy
plt.subplot(3, 2, 4)
for name, (_, _, _, val_accs) in results.items():
    plt.plot(val_accs, label=f"{name} ({param_counts[name]:,} params)")
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Final Accuracy vs Model Size
plt.subplot(3, 2, (5, 6))
sizes = [param_counts[name] for name in model_configs.keys()]
final_train_accs = [results[name][1][-1] for name in model_configs.keys()]
final_val_accs = [results[name][3][-1] for name in model_configs.keys()]

plt.semilogx(sizes, final_train_accs, 'o-', label='Training Accuracy')
plt.semilogx(sizes, final_val_accs, 'o-', label='Validation Accuracy')
plt.title('Final Accuracy vs Model Size')
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('neuron_comparison.png')
plt.close()

print("\nTraining completed! Results saved to neuron_comparison.png") 