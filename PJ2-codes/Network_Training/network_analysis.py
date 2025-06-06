import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import seaborn as sns
from sklearn.manifold import TSNE
import cv2
from network import SimpleCNN, CustomCNN, ModernCNN

def load_model(model_path):
    """Load the trained model"""
    # 首先加载模型参数
    state_dict = torch.load(model_path)
    
    model = CustomCNN()
    
    # 加载参数到模型中
    model.load_state_dict(state_dict)
    model.eval()
    return model

def visualize_filters(model, layer_name=None):
    """Visualize the filters of the first convolutional layer"""
    # Get the first convolutional layer
    if layer_name is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_name = name
                break
    
    conv_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            conv_layer = module
            break
    
    if conv_layer is None:
        print("No convolutional layer found")
        return
    
    # Get the weights
    weights = conv_layer.weight.data.cpu().numpy()
    
    # Normalize the weights for visualization
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Calculate the grid size
    n_filters = weights.shape[0]
    n_channels = weights.shape[1]
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    
    # Create a figure
    plt.figure(figsize=(20, 20))
    
    # Plot each filter
    for i in range(n_filters):
        plt.subplot(grid_size, grid_size, i + 1)
        # For RGB images, show the first channel
        plt.imshow(weights[i, 0], cmap='gray')
        plt.axis('off')
    
    plt.suptitle(f'Filters from {layer_name}')
    plt.savefig('filters_visualization.png')
    plt.close()

def plot_loss_landscape(model, criterion, data_loader, device):
    """Plot the loss landscape around the current model parameters"""
    # Get the current parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Create a grid of parameter perturbations
    n_points = 20
    alphas = np.linspace(-1, 1, n_points)
    betas = np.linspace(-1, 1, n_points)
    
    # Initialize the loss landscape
    loss_landscape = np.zeros((n_points, n_points))
    
    # Calculate the loss for each point in the grid
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb the parameters
            for p in params:
                p.data = p.data + alpha * 0.1 * torch.randn_like(p.data) + beta * 0.1 * torch.randn_like(p.data)
            
            # Calculate the loss
            total_loss = 0
            n_batches = 0
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                n_batches += 1
            
            loss_landscape[i, j] = total_loss / n_batches
    
    # Plot the loss landscape
    plt.figure(figsize=(10, 8))
    sns.heatmap(loss_landscape, cmap='viridis')
    plt.title('Loss Landscape')
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.savefig('loss_landscape.png')
    plt.close()

def visualize_feature_maps(model, input_tensor, layer_name=None):
    """Visualize the feature maps of a specific layer"""
    # Get the layer
    if layer_name is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_name = name
                break
    
    # Register hook to get feature maps
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    # Register the hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
            break
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Get the feature maps
    feature_maps = feature_maps[0]
    
    # Plot the feature maps
    n_maps = feature_maps.shape[1]
    grid_size = int(np.ceil(np.sqrt(n_maps)))
    
    plt.figure(figsize=(20, 20))
    for i in range(n_maps):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name}')
    plt.savefig('feature_maps.png')
    plt.close()

def visualize_all_conv_layers(model):
    """Visualize filters from all convolutional layers in the model"""
    # Find all convolutional layers
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    
    if not conv_layers:
        print("No convolutional layers found")
        return
    
    # Create a figure for each layer
    for layer_name, conv_layer in conv_layers:
        # Get the weights
        weights = conv_layer.weight.data.cpu().numpy()
        
        # Normalize the weights for visualization
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        # Calculate the grid size
        n_filters = weights.shape[0]
        n_channels = weights.shape[1]
        grid_size = int(np.ceil(np.sqrt(n_filters)))
        
        # Create a figure
        plt.figure(figsize=(20, 20))
        
        # Plot each filter
        for i in range(n_filters):
            plt.subplot(grid_size, grid_size, i + 1)
            # For multi-channel filters, show the first channel
            plt.imshow(weights[i, 0], cmap='gray')
            plt.axis('off')
        
        plt.suptitle(f'Filters from {layer_name}\nShape: {weights.shape}')
        plt.savefig(f'filters_{layer_name.replace(".", "_")}.png')
        plt.close()
        
        print(f"Saved visualization for layer: {layer_name}")

def main():
    # Load your model
    model_path = 'Network_Training/results/CustomCNN_best_model.pth'
    model = load_model(model_path)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # 确保默认的tensor类型与MPS兼容
        torch.set_default_tensor_type('torch.FloatTensor')
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Visualize all convolutional layers
    print("Visualizing all convolutional layers...")
    visualize_all_conv_layers(model)
    
    # Note: To visualize loss landscape and feature maps, you'll need to provide:
    # 1. A data loader
    # 2. A loss function
    # 3. Sample input tensor
    # These can be added based on your specific model and dataset

if __name__ == '__main__':
    main() 