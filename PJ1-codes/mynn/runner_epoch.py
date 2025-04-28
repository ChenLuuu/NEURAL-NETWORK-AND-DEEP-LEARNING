import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    RunnerM handles training, evaluating, saving, and loading the model.
    """

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set
            assert X.shape[0] == y.shape[0]

            # Shuffle dataset
            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            num_batches = int(np.ceil(X.shape[0] / self.batch_size))
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

            epoch_train_losses = []
            epoch_train_scores = []

            for iteration in pbar:
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                trn_score = self.metric(logits, train_y)

                epoch_train_losses.append(trn_loss)
                epoch_train_scores.append(trn_score)

                self.loss_fn.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                # 实时更新 tqdm 后缀信息
                pbar.set_postfix({
                    'loss': f'{trn_loss:.4f}',
                    'acc': f'{trn_score:.4f}'
                })

            # After each epoch, evaluate on dev set
            dev_score, dev_loss = self.evaluate(dev_set)

            # Logging
            avg_train_loss = np.mean(epoch_train_losses)
            avg_train_score = np.mean(epoch_train_scores)

            self.train_loss.append(avg_train_loss)
            self.train_scores.append(avg_train_score)
            self.dev_loss.append(dev_loss)
            self.dev_scores.append(dev_score)

            print(f"[Epoch {epoch+1}] Train loss: {avg_train_loss:.4f}, acc: {avg_train_score:.4f} | Val loss: {dev_loss:.4f}, acc: {dev_score:.4f}")

            # Save best model
            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"New best accuracy: {best_score:.5f} → {dev_score:.5f}")
                best_score = dev_score

        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)