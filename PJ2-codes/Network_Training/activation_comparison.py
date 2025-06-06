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
from network import CustomCNN

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
        
        # Create progress bar for this epoch
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, data in enumerate(pbar, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(i+1):.3f}',
                'acc': f'{100*correct/total:.1f}%'
            })
        
        # Evaluate metrics at the end of each epoch
        train_acc, train_loss = evaluate_metrics(model, trainloader, criterion)
        val_acc, val_loss = evaluate_metrics(model, valloader, criterion)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
    
    return train_losses, train_accs, val_losses, val_accs

# Define activation functions to compare
activations = {
    'ReLU': F.relu,
    'LeakyReLU': F.leaky_relu,
    'GELU': F.gelu,
    'Tanh': torch.tanh,
    'Sigmoid': torch.sigmoid
}

# Train models with different activation functions
results = {}
epochs = 10

for name, activation in activations.items():
    print(f"\nTraining model with {name} activation...")
    model = CustomCNN(activation_fn=activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    metrics = train_model(model, optimizer, criterion, epochs=epochs, name=name)
    results[name] = metrics

# Create plots
plt.figure(figsize=(20, 10))

# Training Loss
plt.subplot(2, 2, 1)
for name, (train_losses, _, _, _) in results.items():
    plt.plot(train_losses, label=name)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Training Accuracy
plt.subplot(2, 2, 2)
for name, (_, train_accs, _, _) in results.items():
    plt.plot(train_accs, label=name)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Validation Loss
plt.subplot(2, 2, 3)
for name, (_, _, val_losses, _) in results.items():
    plt.plot(val_losses, label=name)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Validation Accuracy
plt.subplot(2, 2, 4)
for name, (_, _, _, val_accs) in results.items():
    plt.plot(val_accs, label=name)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('activation_comparison.png')
plt.close()

print("\nTraining completed! Results saved to activation_comparison.png") 