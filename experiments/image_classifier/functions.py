from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import glob
from sklearn.model_selection import train_test_split
from os.path import exists


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


def load_data(dir_true, dir_false, normalize=True):
    train_true = datasets.ImageFolder(dir_true)
    train_true_loader = DataLoader(train_true)

    return train_true_loader


# data = load_data("../../datasets/negev/articles_molecules", "../../datasets/negev/not_molecules")
# print(data[0][1])

# def train(model, n_epochs, eta, train_x, train_y, eta):
#
#     for epoch in range(n_epochs):
