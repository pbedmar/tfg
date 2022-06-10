import math
import numpy as np
from functools import partial

from datasets import CompoundDataset
from functions import train, test, dataset_mean_std, init_weights
from models import LeNet5, AlexNet, VGG16

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import KFold


print = partial(print, flush=True)

torch.manual_seed(1)
np.random.seed(1)

nb_epochs = 50
batch_size = 32
folds = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()

models = ["LeNet5", "Alexnet", "VGG16"]
lrs = [5e-1, 5e-2, 5e-3, 5e-4, 5e-5]
weight_initializations = ["He", "Xavier"]
optimizers = ["SGD", "Adam", "Adadelta"]

train_dataset = CompoundDataset(
    "../../datasets/negev/articles_molecules/preprocess256/aug2",
    "../../datasets/negev/not_molecules/preprocess256",
    "train",
    transform=transforms.ToTensor()
)

mean, std = dataset_mean_std(train_dataset)


train_dataset = CompoundDataset(
    "../../datasets/negev/articles_molecules/preprocess256/aug2",
    "../../datasets/negev/not_molecules/preprocess256",
    "train",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean[0], mean[1], mean[2]],
                            [std[0], std[1], std[2]])
    ])
)

test_dataset = CompoundDataset(
    "../../datasets/negev/articles_molecules/preprocess256/aug2",
    "../../datasets/negev/not_molecules/preprocess256",
    "test",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean[0], mean[1], mean[2]],
                             [std[0], std[1], std[2]])
    ])
)

losses = torch.zeros(len(models), len(weight_initializations), len(optimizers), len(lrs), folds, nb_epochs)
results = torch.zeros(len(models), len(weight_initializations), len(optimizers), len(lrs), 4, 4)

for m, model_s in enumerate(models):
    print("")
    for w, weight_initialization_s in enumerate(weight_initializations):
        print("")
        for o, optimizer_s in enumerate(optimizers):
            print("")
            for l, lr in enumerate(lrs):
                print("")
                print("---- MODEL:", model_s, " WEIGHT_INIT:", weight_initialization_s, " OPTIMIZER:", optimizer_s, " LR:", lr, "----")

                cross_val = KFold(n_splits=folds, shuffle=True)
                fold_results = torch.zeros(4, folds)

                for fold, (train_idx, val_idx) in enumerate(cross_val.split(train_dataset)):
                    print("Fold:", fold)

                    torch.manual_seed(fold)
                    np.random.seed(fold)

                    model = eval(model_s + "()")
                    model.to(device)

                    model = model.apply(lambda layer: init_weights(layer, optimizer_s))

                    optimizer = eval("optim."+optimizer_s+"(model.parameters(), lr)")

                    train_subset = torch.utils.data.SubsetRandomSampler(train_idx)
                    val_subset = torch.utils.data.SubsetRandomSampler(val_idx)

                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subset)
                    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subset)

                    losses[m,w,o,l,fold] = train(train_dataloader, model, criterion, nb_epochs, lr, device, optimizer)
                    torch.save(model.state_dict(), "stored_models/"+model_s+"_"+weight_initialization_s+"_"+optimizer_s+"_"+str(lr)+"_"+str(fold)+".pt")

                    nb_train_errors = test(train_dataloader, model, device)
                    nb_test_errors = test(val_dataloader, model, device)

                    fold_results[0, fold] = nb_train_errors
                    fold_results[1, fold] = nb_train_errors / len(train_idx) * 100
                    fold_results[2, fold] = nb_test_errors
                    fold_results[3, fold] = nb_test_errors/len(val_idx)*100


                for i in range(4):
                    results[m, w, o, l, i, 0] = torch.mean(fold_results[i])
                    results[m, w, o, l, i, 1] = torch.std(fold_results[i])
                    results[m, w, o, l, i, 2] = torch.max(fold_results[i])
                    results[m, w, o, l, i, 3] = torch.min(fold_results[i])



                print("Cross validation error mean (train split):", results[m, w, o, l, 0, 0])
                print("Cross validation error stdev (train split):", results[m, w, o, l, 0, 1])
                print("Cross validation error max (train split):", results[m, w, o, l, 0, 2])
                print("Cross validation error min (train split):", results[m, w, o, l, 0, 3])
                print("Cross validation error percentage (%) mean (train split):", results[m, w, o, l, 1, 0], "%")
                print("Cross validation error percentage (%) stdev (train split):", results[m, w, o, l, 1, 1], "%")
                print("Cross validation error percentage (%) max (train split):", results[m, w, o, l, 1, 2], "%")
                print("Cross validation error percentage (%) min (train split):", results[m, w, o, l, 1, 3], "%")
                print("Cross validation error mean (val split):", results[m, w, o, l, 2, 0])
                print("Cross validation error stdev (val split):", results[m, w, o, l, 2, 1])
                print("Cross validation error max (val split):", results[m, w, o, l, 2, 2])
                print("Cross validation error min (val split):", results[m, w, o, l, 2, 3])
                print("Cross validation error percentage (%) mean (val split):", results[m, w, o, l, 3, 0], "%")
                print("Cross validation error percentage (%) stdev (val split):", results[m, w, o, l, 3, 1], "%")
                print("Cross validation error percentage (%) max (val split):", results[m, w, o, l, 3, 2], "%")
                print("Cross validation error percentage (%) min (val split):", results[m, w, o, l, 3, 3], "%")

torch.save(losses, "stored_results/grid_search_losses.pt")
torch.save(results, "stored_results/grid_search_results.pt")