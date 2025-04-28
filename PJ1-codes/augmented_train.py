import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import cv2
import mynn as nn
from draw_tools.plot import plot

# fixed seed for experiment
np.random.seed(309)

def augment_image(image, max_translation=2, max_rotation=10, max_scale=0.1):
    """
    Apply random transformations to an image
    Args:
        image: 28x28 numpy array
        max_translation: maximum pixels to translate
        max_rotation: maximum degrees to rotate
        max_scale: maximum scale factor
    Returns:
        augmented image
    """
    # Reshape to 28x28
    img = image.reshape(28, 28)
    
    # Random translation
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (28, 28))
    
    # Random rotation
    angle = np.random.uniform(-max_rotation, max_rotation)
    M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    img = cv2.warpAffine(img, M, (28, 28))
    
    # Random scaling
    scale = 1 + np.random.uniform(-max_scale, max_scale)
    M = cv2.getRotationMatrix2D((14, 14), 0, scale)
    img = cv2.warpAffine(img, M, (28, 28))
    
    return img.reshape(-1)

def load_mnist_data():
    train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
    train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)
    
    return train_imgs, train_labs

def create_augmented_dataset(images, labels, augmentation_factor=2):
    """
    Create augmented dataset by applying transformations
    Args:
        images: original images
        labels: original labels
        augmentation_factor: how many times to augment each image
    Returns:
        augmented images and labels
    """
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        # Add original image
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Add augmented versions
        for _ in range(augmentation_factor):
            aug_img = augment_image(img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)

def main():
    # Load original data
    train_imgs, train_labs = load_mnist_data()
    
    # Create augmented dataset
    train_imgs, train_labs = create_augmented_dataset(train_imgs, train_labs, augmentation_factor=2)
    
    # Choose 10000 samples from train set as validation set
    idx = np.random.permutation(np.arange(len(train_imgs)))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
    
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    # Normalize from [0, 255] to [0, 1]
    train_imgs = train_imgs / train_imgs.max()
    valid_imgs = valid_imgs / valid_imgs.max()

    # Create and train model
    linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

    runner = nn.runner_epoch.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, 
                                   scheduler=scheduler, batch_size=32)
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], 
                num_epochs=5, log_iters=100, save_dir=r'./best_models')

    # Plot results
    _, axes = plt.subplots(1, 2)
    axes.reshape(-1)
    _.set_tight_layout(1)
    plot(runner, axes)
    plt.show()

if __name__ == "__main__":
    main() 