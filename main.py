import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import SeqDataSet
from model import SeqClassifyModel
from preprocess import load_data

torch.set_default_tensor_type('torch.DoubleTensor')

def dataloader(data, label, num_class):
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.33)

    train_loader = DataLoader(
        SeqDataSet(X_train, Y_train, num_class),
        batch_size=1,
        shuffle=True
    )

    test_loader = DataLoader(
        SeqDataSet(seq_data=X_test, label=Y_test, num_class=num_class),
        batch_size=1,
        shuffle=True
    )

    return train_loader, test_loader


def main():
    file_path = '/home/bigheiniu/course/jj/167_6.csv'
    data, label = load_data(file_path)
    train_loader, test_loader = dataloader(data, label, num_class=3)
    model = SeqClassifyModel(in_chanell=1, out_chanell=32, kenel_size=(3,3), num_class=3)
    loss_fn = torch.nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    epoch_count = 20
    for i in range(epoch_count):
        train_epoch(model, train_loader, loss_fn, optim)
        test_epoch(model, test_loader, loss_fn)
        print("\n[INFO] finish epoch {}".format(i))

def train_epoch(model, train_loader, loss_fn, optim):
    model.train()
    for batch in train_loader:
        # tuple, (seq, label)
        # batch stands for data from __getitem__
        seq, label = batch

        result = model(seq)
        result = torch.sigmoid(result)
        optim.zero_grad()
        loss = loss_fn(result, label)
        print("[INFO] Train Loss {}".format(loss.item()))
        loss.backward()
        optim.step()

def test_epoch(model, test_loader, loss_fn):
    model.eval()
    for batch in test_loader:
        seq, label = batch
        result = model(seq)
        result = torch.sigmoid(result)
        loss = loss_fn(result, label)
        print("[INFO] Test loss {}".format(loss.item()))


if __name__ == '__main__':
    main()







