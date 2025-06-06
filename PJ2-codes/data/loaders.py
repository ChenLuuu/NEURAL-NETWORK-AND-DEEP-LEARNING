"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets



class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(train=True, download=False):
    root = './data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=train,
        num_workers=0  # Set to 0 to disable multiprocessing
    )

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break