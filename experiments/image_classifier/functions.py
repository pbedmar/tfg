import glob
import math

import torch
from torch import optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

def gen_metadata(directory, seed=1):
    directory = directory+"/"
    filenames = glob.glob(directory+"*.jpg")
    train, test = train_test_split(filenames, test_size=0.15, random_state=seed)

    with open(directory+"imclass_train.txt", "w") as f:
        for filename in train[:-1]:
            f.write(filename+"\n")
        f.write(train[-1])

    with open(directory+"imclass_test.txt", "w") as f:
        for filename in test[:-1]:
            f.write(filename+"\n")
        f.write(test[-1])


def train(dataloader, model, criterion, nb_epochs, lr, device, optimizer):
    acc_losses_by_epoch = torch.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        acc_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            acc_loss = acc_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

        acc_losses_by_epoch[epoch] = acc_loss
        if epoch % 10 == 0:
            print("Epoch nb.", epoch, "-> loss=",acc_loss)

    return acc_losses_by_epoch


def test(dataset, model, device):
    nb_errors = 0

    for batch_x, batch_y in dataset:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)

        _, predicted_classes = output.max(1)

        for predicted_class, y in zip(predicted_classes, batch_y):
            if predicted_class != y:
                nb_errors = nb_errors + 1

    return nb_errors


def dataset_mean_std(dataset):
    train_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(train_dataloader))[0]
    mean = torch.mean(data, dim=(0,2,3))
    std = torch.std(data, dim=(0,2,3))

    return mean, std


def init_weights(layer, optimizer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
        if optimizer == "Adam":
            torch.nn.init.xavier_uniform_(layer.weight)
        else:
            torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

