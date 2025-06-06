"""
VGG Loss Landscape Analysis
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
import torch.multiprocessing as mp

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device:", device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using device:", device)

# Initialize data loaders
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device == 'cuda':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=10, best_model_path=None):
    model.to(device)
    learning_curve = []  # Record average loss per epoch
    train_accuracy_curve = []  # Record training accuracy per epoch
    val_accuracy_curve = []  # Record validation accuracy per epoch
    losses_list = []  # Record all losses during training
    grads = []  # Record all gradients during training

    print(f"\nTraining for {epochs_n} epochs...")
    for epoch in range(epochs_n):
        if scheduler is not None:
            scheduler.step()
        model.train()
        epoch_loss = 0
        epoch_losses = []  # Record losses for this epoch
        epoch_grads = []  # Record gradients for this epoch
        
        print(f"\nEpoch [{epoch+1}/{epochs_n}]")
        pbar = tqdm(train_loader, desc=f'Training', leave=True)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Record loss
            epoch_loss += loss.item()
            epoch_losses.append(loss.item())
            current_loss = epoch_loss / (batch_idx + 1)
            
            loss.backward()
            
            # Record gradient (from the last linear layer)
            if isinstance(model, VGG_A) or isinstance(model, VGG_A_BatchNorm):
                grad = model.classifier[-1].weight.grad.norm().item()
                epoch_grads.append(grad)
            
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}'
            })
        
        # Calculate average loss and accuracy for this epoch
        avg_loss = epoch_loss / len(train_loader)
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        
        # Record metrics
        learning_curve.append(avg_loss)
        train_accuracy_curve.append(train_acc)
        val_accuracy_curve.append(val_acc)
        losses_list.append(epoch_losses)
        grads.append(epoch_grads)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs_n} Summary:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')

    return losses_list, grads, learning_curve, train_accuracy_curve, val_accuracy_curve

if __name__ == "__main__":
    # Create directory for figures if it doesn't exist
    os.makedirs('VGG_BatchNorm/figs', exist_ok=True)
    
    # Train both models
    epo = 20

    print("Training VGG_A without BatchNorm")
    # Train VGG_A without BatchNorm
    set_random_seeds(seed_value=2020, device=device.type)
    model_no_bn = VGG_A()
    optimizer_no_bn = torch.optim.Adam(model_no_bn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    results_no_bn = train(model_no_bn, optimizer_no_bn, criterion, train_loader, val_loader, epochs_n=epo)
    losses_no_bn, grads_no_bn, learning_curve_no_bn, train_acc_no_bn, val_acc_no_bn = results_no_bn

    print("Training VGG_A with BatchNorm")
    # Train VGG_A with BatchNorm
    set_random_seeds(seed_value=2020, device=device.type)
    model_bn = VGG_A_BatchNorm()
    optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=0.001)
    results_bn = train(model_bn, optimizer_bn, criterion, train_loader, val_loader, epochs_n=epo)
    losses_bn, grads_bn, learning_curve_bn, train_acc_bn, val_acc_bn = results_bn

    # Plot final comparison
    plt.figure(figsize=(15, 5))

    # Plot training loss comparison
    plt.subplot(131)
    plt.plot(learning_curve_no_bn, label='Without BN')
    plt.plot(learning_curve_bn, label='With BN')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy comparison
    plt.subplot(132)
    plt.plot(train_acc_no_bn, label='Train (No BN)')
    plt.plot(val_acc_no_bn, label='Val (No BN)')
    plt.plot(train_acc_bn, label='Train (BN)')
    plt.plot(val_acc_bn, label='Val (BN)')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot average gradient norm per epoch
    plt.subplot(133)
    epoch_grad_means_no_bn = [np.mean(epoch_grads) for epoch_grads in grads_no_bn]
    epoch_grad_means_bn = [np.mean(epoch_grads) for epoch_grads in grads_bn]
    plt.plot(epoch_grad_means_no_bn, label='Without BN')
    plt.plot(epoch_grad_means_bn, label='With BN')
    plt.title('Average Gradient Norm per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()

    plt.tight_layout()
    plt.savefig('VGG_BatchNorm/figs/training_summary.png')
    plt.close()

    # Plot loss landscape
    def plot_loss_landscape():
        plt.figure(figsize=(10, 6))
        
        # Plot losses for model without BatchNorm
        all_losses_no_bn = []
        for epoch_losses in losses_no_bn:
            all_losses_no_bn.extend(epoch_losses)
        plt.plot(all_losses_no_bn, 'b-', alpha=0.5, label='Without BatchNorm')
        
        # Plot losses for model with BatchNorm
        all_losses_bn = []
        for epoch_losses in losses_bn:
            all_losses_bn.extend(epoch_losses)
        plt.plot(all_losses_bn, 'r-', alpha=0.5, label='With BatchNorm')
        
        plt.title('Loss Landscape During Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('VGG_BatchNorm/figs/loss_landscape.png')
        plt.close()

    # Plot the loss landscape
    plot_loss_landscape()