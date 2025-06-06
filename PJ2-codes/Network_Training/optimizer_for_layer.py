import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from network import CustomCNN
from resnet import ResNet18
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
from torchinfo import summary

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

# Initialize network
net = CustomCNN()
net = net.to(device)

# Print model summary
print("\nModel Summary:")
summary(net.to('cpu'), input_size=(1, 3, 32, 32),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=20,
        row_settings=["var_names"])
net = net.to(device)

# Load datasets
print("\nLoading datasets...")
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
train_size = int(0.99 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])
print(f"Dataset sizes - Train: {len(trainset)}, Validation: {len(valset)}, Test: {len(testset)}")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Criterion
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ----------- Custom Optimizer Section ------------

# Separate parameter groups
conv_params = list(net.conv1.parameters()) + list(net.conv2.parameters()) + list(net.conv3.parameters())
fc_params = list(net.fc1.parameters()) + list(net.fc2.parameters())

# Optimizer for fully connected layers
optimizer_fc = optim.AdamW(fc_params, lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fc, T_max=50)

# Custom SGD step
def custom_sgd_step(params, lr=0.01, momentum=0.9, velocity_dict=None):
    if velocity_dict is None:
        velocity_dict = {}
    for p in params:
        if p.grad is not None:
            if p not in velocity_dict:
                velocity_dict[p] = torch.zeros_like(p.data)
            v = velocity_dict[p]
            v.mul_(momentum).add_(p.grad.data)
            p.data -= lr * v
    return velocity_dict

# ----------- Evaluation Utilities ------------
def evaluate_accuracy(model, dataloader):
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

# ----------- Training Function ------------
def train(epochs=10):
    iteration_losses = []
    iterations = []
    total_iterations = 0

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    epochs_list = []
    best_val_acc = 0.0
    velocity_dict = {}

    print("\nStarting training...")
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')

        for i, data in enumerate(pbar, 0):
            total_iterations += 1
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero gradients for both groups
            optimizer_fc.zero_grad()
            for p in conv_params:
                if p.grad is not None:
                    p.grad.zero_()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Optimizer steps
            optimizer_fc.step()
            velocity_dict = custom_sgd_step(conv_params, lr=0.01, momentum=0.9, velocity_dict=velocity_dict)

            # Logging
            running_loss += loss.item()
            iteration_losses.append(loss.item())
            iterations.append(total_iterations)
            pbar.set_postfix({'loss': f'{running_loss/(i+1):.3f}'})

        scheduler.step()

        # Metrics
        train_acc, train_loss = evaluate_metrics(net, trainloader, criterion)
        val_acc, val_loss = evaluate_metrics(net, valloader, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        epochs_list.append(epoch + 1)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), 'best_model.pth')

    return epochs_list, train_losses, train_accuracies, val_losses, val_accuracies

# ----------- Start Training ------------
epochs_list, train_losses, train_accuracies, val_losses, val_accuracies = train(epochs=20)

# Load best model and evaluate on test set
net.load_state_dict(torch.load('best_model.pth'))
test_accuracy = evaluate_accuracy(net, testloader)
print(f'\nFinal test accuracy: {test_accuracy:.2f}%')
print('Finished Training')

# ----------- Plot Results ------------
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(epochs_list, train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(epochs_list, train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.subplot(2, 2, 3)
plt.plot(epochs_list, val_losses)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 4)
plt.plot(epochs_list, val_accuracies)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig('Training_Results.png')
plt.close()