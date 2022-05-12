import glob

import torch
from torch import optim

from sklearn.model_selection import train_test_split

def gen_metadata(directory):
    directory = directory+"/"
    filenames = glob.glob(directory+"*.jpg")
    train, test = train_test_split(filenames, test_size=0.25, random_state=42)

    with open(directory+"imclass_train.txt", "w") as f:
        for filename in train[:-1]:
            f.write(filename+"\n")
        f.write(train[-1])

    with open(directory+"imclass_test.txt", "w") as f:
        for filename in test[:-1]:
            f.write(filename+"\n")
        f.write(test[-1])


def train(dataloader, model, criterion, nb_epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr)

    acc_losses_by_epoch = torch.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        print("Epoch nb.",epoch)
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
        print("loss=", acc_loss)
        print()

    return acc_losses_by_epoch