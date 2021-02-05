import time

import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_handler import get_data, split_data

EPOCHS = 20


class FFNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FFNet, self).__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.l1 = nn.Linear(num_features, 64)
        self.l2 = nn.Linear(64, num_classes)
        self.do = nn.Dropout(0.1)
        self.z = None

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = self.l1(x)
        x = F.relu(x)
        # z = x
        self.z = x
        x = self.do(x)
        x = self.l2(x)  # Num_classes
        out = F.log_softmax(x, dim=1)
        return out

    def predict_proba(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        return np.exp(self.forward(x).detach().cpu().numpy())
        # return torch.exp(self.forward(x))

    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        return np.argmax(self.forward(x).detach().cpu().numpy(), axis=1)
        # return torch.argmax(self.forward(x), dim=1)

    def fit(self, X, y):
        train_ann(self, X, y, EPOCHS)


def train_ann(model, X, y, epochs):
    model.train()
    n = len(X)
    bs = 16
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(epochs):
        y = np.expand_dims(y, 1)
        data = np.hstack([X, y])
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1]
        losses = []
        for b in range(int(n / bs)):
            idx_start = b * bs
            idx_end = min((b + 1) * bs, n)
            batch_X = torch.tensor(X[idx_start:idx_end], dtype=torch.float32).to(model.device)  # .cuda()
            batch_y = torch.tensor(y[idx_start:idx_end], dtype=torch.long).to(model.device)  # .cuda()

            model.zero_grad()
            output = model(batch_X)
            loss = F.nll_loss(output, batch_y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        # print("Epoch", epoch, "mean loss over batches:", np.mean(losses))


def stuff():
    train, val = get_data("Skewed")
    X_train, y_train, X_val, y_val = split_data(train, val)
    num_features = X_train.shape[1]
    num_classes = int(max(y_train) + 1)
    # print(num_features, num_classes)
    t1 = time.time()
    model = FFNet(num_features, num_classes)
    model = model.to(model.device)
    # train_ann(model, X_train, y_train, EPOCHS)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)
    # print(np.sum(probs, axis=1))
    y_pred = model.predict(X_val)
    f1 = sklearn.metrics.f1_score(y_val, y_pred, average="macro")
    t2 = time.time()
    print("Time taken:", t2-t1)
    print(f1)


# Cuda: add cuda() to model, data to cuda

if __name__ == '__main__':
    stuff()
