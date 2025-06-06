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
import copy

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128
learning_rates = [1e-4, 2e-4, 5e-4, 1e-3]  # Different learning rates to test

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"\nUsing device: {device}")

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

def train_multiple_lr(model_class, train_loader, val_loader, learning_rates, epochs_n=10):
    """Train multiple models with different learning rates and track their losses"""
    all_losses = []  # List to store losses for each learning rate
    models = []      # List to store trained models
    
    model_name = "VGG-A with BatchNorm" if model_class == VGG_A_BatchNorm else "VGG-A without BatchNorm"
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    for lr in learning_rates:
        print(f"\n{'-'*50}")
        print(f"Model: {model_name}")
        print(f"Learning Rate: {lr}")
        print(f"{'-'*50}\n")
        
        # Create model and silently move to device
        model = model_class()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model_losses = []  # Store losses for this specific learning rate
        
        for epoch in range(epochs_n):
            print(f"\nEpoch [{epoch+1}/{epochs_n}]")
            model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f'Training (lr={lr})', leave=True)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            model_losses.extend(epoch_losses)
        
        all_losses.append(model_losses)
        models.append(model)
    
    return all_losses, models

def plot_loss_landscape_comparison():
    """Plot loss landscape comparison between models with and without BatchNorm"""
    # Create directory for figures if it doesn't exist
    os.makedirs('VGG_BatchNorm/figs', exist_ok=True)
    
    # Train models without BatchNorm
    print("Training VGG_A without BatchNorm for different learning rates")
    set_random_seeds(seed_value=2020, device=device.type)
    losses_no_bn, models_no_bn = train_multiple_lr(VGG_A, train_loader, val_loader, learning_rates, epochs_n=epo)
    
    # Train models with BatchNorm
    print("\nTraining VGG_A with BatchNorm for different learning rates")
    set_random_seeds(seed_value=2020, device=device.type)
    losses_bn, models_bn = train_multiple_lr(VGG_A_BatchNorm, train_loader, val_loader, learning_rates, epochs_n=epo)
    
    # Calculate min and max curves for both models
    steps = min(len(losses_no_bn[0]), len(losses_bn[0]))
    
    # Process no BatchNorm results
    min_curve_no_bn = []
    max_curve_no_bn = []
    for step in range(steps):
        step_losses = [losses[step] for losses in losses_no_bn if step < len(losses)]
        min_curve_no_bn.append(min(step_losses))
        max_curve_no_bn.append(max(step_losses))
    
    # Process BatchNorm results
    min_curve_bn = []
    max_curve_bn = []
    for step in range(steps):
        step_losses = [losses[step] for losses in losses_bn if step < len(losses)]
        min_curve_bn.append(min(step_losses))
        max_curve_bn.append(max(step_losses))
    
    # Plot the combined loss landscape
    plt.figure(figsize=(12, 6))
    
    # Plot for model without BatchNorm
    plt.fill_between(range(steps), min_curve_no_bn, max_curve_no_bn, 
                    alpha=0.3, color='red', label='Without BN Range')
    
    # Plot for model with BatchNorm
    plt.fill_between(range(steps), min_curve_bn, max_curve_bn, 
                    alpha=0.3, color='blue', label='With BN Range')
    
    plt.title('Loss Landscape Comparison (Different Learning Rates)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('VGG_BatchNorm/figs/loss_landscape_comparison.png')
    plt.close()

if __name__ == "__main__":
    epo = 3
    # Plot the loss landscape comparison
    plot_loss_landscape_comparison()