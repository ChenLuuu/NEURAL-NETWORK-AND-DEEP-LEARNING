import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from network import SimpleCNN, CustomCNN, ModernCNN
from densenet import DenseNet121, DenseNetBC100, MyDenseNetBC100, MyDenseNetBC100_Mish
from resnet import ResNet18
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
from torchinfo import summary

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # 确保默认的tensor类型与MPS兼容
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize network and print summary
# net = ModernCNN()

# net = MyDenseNetBC100_Mish()
net = MyDenseNetBC100()
net = net.to(device)

# 使用CPU进行模型总结以避免MPS的兼容性问题
print("\nModel Summary:")
summary(net.to('cpu'), input_size=(1, 3, 32, 32), 
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=20,
        row_settings=["var_names"])

# 确保模型在正确的设备上
net = net.to(device)

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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Optimizer and criterion
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

def evaluate_accuracy(model, dataloader):
    """
    Evaluate model accuracy on the provided dataset
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

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

# Training function
def train(epochs=10):
    iteration_losses = []
    iterations = []
    total_iterations = 0
    
    # Lists to store epoch-wise metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs_list = []
    
    best_val_acc = 0.0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        
        # Create progress bar for this epoch
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, data in enumerate(pbar, 0):
            total_iterations += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Record loss for every iteration
            iteration_losses.append(loss.item())
            iterations.append(total_iterations)
            
            # Update progress bar description
            pbar.set_postfix({
                'loss': f'{running_loss/(i+1):.3f}'
            })
        
        # Evaluate metrics at the end of each epoch
        train_acc, train_loss = evaluate_metrics(net, trainloader, criterion)
        val_acc, val_loss = evaluate_metrics(net, valloader, criterion)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        epochs_list.append(epoch + 1)
        
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
            
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), 'best_model.pth')
    
    return epochs_list, train_losses, train_accuracies, val_losses, val_accuracies

# Train the network
epochs_list, train_losses, train_accuracies, val_losses, val_accuracies = train(epochs=20)

# Load best model and evaluate on test set
net.load_state_dict(torch.load('best_model.pth'))
test_accuracy = evaluate_accuracy(net, testloader)
print(f'\nFinal test accuracy: {test_accuracy:.2f}%')
print('Finished Training')

# Plot training metrics
plt.figure(figsize=(15, 10))

# Plot training loss
plt.subplot(2, 2, 1)
plt.plot(epochs_list, train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot training accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs_list, train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

# Plot validation loss
plt.subplot(2, 2, 3)
plt.plot(epochs_list, val_losses)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs_list, val_accuracies)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig('Training_Results.png')
plt.close()