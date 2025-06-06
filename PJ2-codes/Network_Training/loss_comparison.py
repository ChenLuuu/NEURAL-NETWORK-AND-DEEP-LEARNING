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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

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

def train_model(model, optimizer, criterion, l1_lambda=0, l2_lambda=0, epochs=10, name="Model"):
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
            
            # Calculate main loss
            loss = criterion(outputs, labels)
            
            # Add L1 regularization
            if l1_lambda > 0:
                l1_reg = torch.tensor(0., requires_grad=True)
                for param in model.parameters():
                    l1_reg = l1_reg + torch.norm(param, 1)
                loss = loss + l1_lambda * l1_reg
            
            # Add L2 regularization (if not already in optimizer)
            if l2_lambda > 0:
                l2_reg = torch.tensor(0., requires_grad=True)
                for param in model.parameters():
                    l2_reg = l2_reg + torch.norm(param, 2)
                loss = loss + l2_lambda * l2_reg
            
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

class MSEWithSoftmax(nn.Module):
    def __init__(self):
        super(MSEWithSoftmax, self).__init__()
        
    def forward(self, inputs, targets):
        softmax_outputs = F.softmax(inputs, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        return F.mse_loss(softmax_outputs, target_one_hot)

# class KLDivWithSoftmax(nn.Module):
#     def __init__(self, T=1.0):
#         super(KLDivWithSoftmax, self).__init__()
#         self.T = T
        
#     def forward(self, inputs, targets):
#         log_softmax_inputs = F.log_softmax(inputs / self.T, dim=1)
#         target_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
#         return F.kl_div(log_softmax_inputs, target_one_hot, reduction='batchmean') * (self.T ** 2)

# Calculate class weights for Weighted Cross Entropy
def calculate_class_weights(dataset):
    class_counts = torch.zeros(10)
    for _, label in dataset:
        class_counts[label] += 1
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    return class_weights.to(device)

# Get class weights for weighted cross entropy
class_weights = calculate_class_weights(full_trainset)

# Define different loss configurations
loss_configs = {
    'Standard CrossEntropy': {
        'criterion': nn.CrossEntropyLoss(),
        'l1_lambda': 0,
        'l2_lambda': 0
    },
    'Weighted CrossEntropy': {
        'criterion': nn.CrossEntropyLoss(weight=class_weights),
        'l1_lambda': 0,
        'l2_lambda': 0
    },
    'MSE + Softmax': {
        'criterion': MSEWithSoftmax(),
        'l1_lambda': 0,
        'l2_lambda': 0
    },
    # 'KL Divergence': {
    #     'criterion': KLDivWithSoftmax(T=1.0),
    #     'l1_lambda': 0,
    #     'l2_lambda': 0
    # },
    'Label Smoothing': {
        'criterion': LabelSmoothingLoss(classes=10, smoothing=0.1),
        'l1_lambda': 0,
        'l2_lambda': 0
    },
    'CrossEntropy + L1': {
        'criterion': nn.CrossEntropyLoss(),
        'l1_lambda': 1e-5,
        'l2_lambda': 0
    },
    'CrossEntropy + L2': {
        'criterion': nn.CrossEntropyLoss(),
        'l1_lambda': 0,
        'l2_lambda': 1e-4
    }
}

# Train models with different loss functions
results = {}
epochs = 10

for name, config in loss_configs.items():
    print(f"\nTraining model with {name}...")
    model = CustomCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics = train_model(
        model, 
        optimizer, 
        config['criterion'],
        l1_lambda=config['l1_lambda'],
        l2_lambda=config['l2_lambda'],
        epochs=epochs,
        name=name
    )
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
plt.savefig('loss_comparison.png')
plt.close()

print("\nTraining completed! Results saved to loss_comparison.png") 