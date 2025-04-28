# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot
from hyperparameter_search import CNN_ARCHITECTURE, TRAINING_PARAMS, OPTIMIZER_PARAMS, SCHEDULER_PARAMS, MLP_ARCHITECTURE

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(TRAINING_PARAMS['random_seed'])

train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]


train_imgs = train_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28).astype(np.float32) / 255.

# MLP
# linear_model = nn.models.Model_MLP(
#     [train_imgs.shape[-1]] + MLP_ARCHITECTURE['hidden_dims'],
#     MLP_ARCHITECTURE['activation'],
#     MLP_ARCHITECTURE['weight_decay'],
#     dropout_p=0.5  # Add dropout with probability 0.5
# )
# l2_reg = nn.op.L2Regularization(lambda_reg=MLP_ARCHITECTURE['l2_reg'])
# l2_reg.set_model(linear_model)
# linear_model.layers.append(l2_reg)

# CNN
# 定义CNN的结构
layers = [
    nn.op.conv2D(**CNN_ARCHITECTURE['conv1']),
    nn.op.ReLU(),
    nn.op.MaxPool2D(kernel_size=2, stride=2),
    
    nn.op.conv2D(**CNN_ARCHITECTURE['conv2']),
    nn.op.ReLU(),
    nn.op.MaxPool2D(kernel_size=2, stride=2),
    
    nn.op.Flatten(),
    nn.op.Linear(**CNN_ARCHITECTURE['linear1']),
    nn.op.ReLU(),
    nn.op.Linear(**CNN_ARCHITECTURE['linear2'])
]

cnn_model = nn.models.Model_CNN(layers=layers)

# 使用 MomentumGD 优化器
optimizer = nn.optimizer.MomentGD(
    init_lr=OPTIMIZER_PARAMS['init_lr'],
    model=cnn_model,
    mu=OPTIMIZER_PARAMS['momentum']
)

scheduler = nn.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=SCHEDULER_PARAMS['milestones'],
    gamma=SCHEDULER_PARAMS['gamma']
)

loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

# 这是一个epoch runner，每个epoch评估一次
runner = nn.runner_epoch.RunnerM(
    cnn_model,
    optimizer,
    nn.metric.accuracy,
    loss_fn,
    scheduler=scheduler,
    batch_size=TRAINING_PARAMS['batch_size']
)

runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=TRAINING_PARAMS['num_epochs'],
    log_iters=TRAINING_PARAMS['log_iters'],
    save_dir=r'./best_models'
)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

# plt.savefig('figs/result.png')
plt.show()

# 这是上述未注释的代码的训练结果
# Epoch 1/5: 100%|█| 196/196 [31:45<00:00,  9.72s/it, Train loss=0.33, Train acc=0.88, Dev loss=0.28, 
# best accuracy performence has been updated: 0.00000 --> 0.91070
# Epoch 2/5: 100%|█| 196/196 [31:38<00:00,  9.68s/it, Train loss=0.34, Train acc=0.91, Dev loss=0.18, 
# best accuracy performence has been updated: 0.91070 --> 0.94190
# Epoch 3/5: 100%|█| 196/196 [31:55<00:00,  9.77s/it, Train loss=0.11, Train acc=0.94, Dev loss=0.18, 
# best accuracy performence has been updated: 0.94190 --> 0.94520
# Epoch 4/5: 100%|█| 196/196 [31:32<00:00,  9.65s/it, Train loss=0.11, Train acc=0.97, Dev loss=0.14, 
# best accuracy performence has been updated: 0.94520 --> 0.95610
# Epoch 5/5: 100%|█| 196/196 [31:42<00:00,  9.71s/it, Train loss=0.04, Train acc=0.99, Dev loss=0.12, 
# best accuracy performence has been updated: 0.95610 --> 0.96390