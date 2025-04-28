import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.utils import download_url
import numpy as np
from struct import unpack
import gzip
import pickle
import matplotlib.pyplot as plt

from draw_tools.plot import plot  

np.random.seed(309)
torch.manual_seed(309)

def load_mnist_data(img_path, label_path):
    with gzip.open(img_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    with gzip.open(label_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs, labels

train_imgs, train_labs = load_mnist_data(
    './dataset/MNIST/train-images-idx3-ubyte.gz',
    './dataset/MNIST/train-labels-idx1-ubyte.gz'
)

# random train test split
idx = np.random.permutation(np.arange(len(train_labs)))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs, train_labs = train_imgs[idx], train_labs[idx]

valid_imgs = train_imgs[:10000].reshape(-1, 1, 28, 28)
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:].reshape(-1, 1, 28, 28)
train_labs = train_labs[10000:]


train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0

train_dataset = TensorDataset(torch.tensor(train_imgs, dtype=torch.float32), torch.tensor(train_labs))
valid_dataset = TensorDataset(torch.tensor(valid_imgs, dtype=torch.float32), torch.tensor(valid_labs))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=50)


# ===== Model Definition =====
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)




# ===== 3. Runner =====
class Runner:
    def __init__(self, model, optimizer, scheduler, criterion, metric):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metric = metric
        self.logs = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}

    def evaluate(self, loader):
        self.model.eval()
        loss_total, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                out = self.model(x)
                loss = self.criterion(out, y)
                loss_total += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        return loss_total / total, correct / total

    def train(self, train_loader, valid_loader, num_epochs=10, log_iters=10, save_dir='./best_models'):
        best_acc = 0
        for epoch in range(num_epochs):
            self.model.train()
            running_loss, correct, total = 0, 0, 0
            for i, (x, y) in enumerate(train_loader):
                out = self.model(x)
                loss = self.criterion(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)

                if (i + 1) % log_iters == 0:
                    print(f"Epoch {epoch+1}, Iter {i+1}, Loss: {loss.item():.4f}")

            train_loss = running_loss / total
            train_acc = correct / total
            val_loss, val_acc = self.evaluate(valid_loader)
            self.scheduler.step()

            self.logs['train_loss'].append(train_loss)
            self.logs['valid_loss'].append(val_loss)
            self.logs['train_acc'].append(train_acc)
            self.logs['valid_acc'].append(val_acc)

            print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f"{save_dir}/best_model.pth")

# ===== 4. 实例化并训练 =====
model = CNN()  
optimizer = optim.SGD(model.parameters(), lr=0.06, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 2400, 4000], gamma=0.5)
criterion = nn.CrossEntropyLoss()

runner = Runner(model, optimizer, scheduler, criterion, metric='accuracy')
runner.train(train_loader, valid_loader, num_epochs=10, log_iters=100, save_dir='./best_models')
