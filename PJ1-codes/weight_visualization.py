# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# Load the model
layers = [
    nn.op.conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
    nn.op.ReLU(),
    nn.op.MaxPool2D(kernel_size=2, stride=2),
    
    nn.op.conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.op.ReLU(),
    nn.op.MaxPool2D(kernel_size=2, stride=2),
    
    nn.op.Flatten(),
    nn.op.Linear(in_dim=784, out_dim=100),
    nn.op.ReLU(),
    nn.op.Linear(in_dim=100, out_dim=10)
]

model = nn.models.Model_CNN(layers=layers)
model.load_model(r'./best_models/CNN.pickle')

# Visualize first convolutional layer weights
first_conv_weights = model.layers[0].params['W']
n_filters = first_conv_weights.shape[0]

# Create a figure for the first conv layer
plt.figure(figsize=(12, 6))
plt.suptitle('First Convolutional Layer Filters (3x3)')
for i in range(n_filters):
    plt.subplot(2, 4, i+1)
    plt.imshow(first_conv_weights[i, 0], cmap='gray')
    plt.title(f'Filter {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('first_conv_filters.png')
plt.show()

# Visualize second convolutional layer weights
second_conv_weights = model.layers[3].params['W']
n_filters = second_conv_weights.shape[0]

# Create a figure for the second conv layer
plt.figure(figsize=(15, 10))
plt.suptitle('Second Convolutional Layer Filters (3x3)')
for i in range(n_filters):
    plt.subplot(4, 4, i+1)
    # Average across input channels for visualization
    avg_filter = np.mean(second_conv_weights[i], axis=0)
    plt.imshow(avg_filter, cmap='gray')
    plt.title(f'Filter {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('second_conv_filters.png')
plt.show()

# Visualize first fully connected layer weights
fc1_weights = model.layers[7].params['W']
plt.figure(figsize=(12, 8))
plt.imshow(fc1_weights, cmap='viridis')
plt.colorbar()
plt.title('First Fully Connected Layer Weights')
plt.xlabel('Output Neurons')
plt.ylabel('Input Neurons')
plt.savefig('fc1_weights.png')
plt.show()

# Visualize second fully connected layer weights
fc2_weights = model.layers[9].params['W']
plt.figure(figsize=(10, 6))
plt.imshow(fc2_weights, cmap='viridis')
plt.colorbar()
plt.title('Second Fully Connected Layer Weights')
plt.xlabel('Output Classes')
plt.ylabel('Input Neurons')
plt.savefig('fc2_weights.png')
plt.show()

test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

# Visualize first conv layer filters in a grid
plt.figure(figsize=(15, 15))
for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.imshow(first_conv_weights[i, 0], cmap='gray')
    plt.title(f'Filter {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('conv_filters_grid.png')
plt.show()

# Visualize first fully connected layer weights
plt.figure(figsize=(15, 15))
plt.imshow(fc1_weights, cmap='viridis')
plt.colorbar()
plt.title('First Fully Connected Layer Weights')
plt.xlabel('Output Neurons')
plt.ylabel('Input Neurons')
plt.savefig('fc1_weights_large.png')
plt.show()