import numpy as np

from datasets import CompoundDataset
from functions import train, test
from models import LeNet5, AlexNet, VGG16

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import KFold

torch.manual_seed(0)

nb_epochs = 70
batch_size = 32
folds = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()

models = ["LeNet5"]
lrs = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]

train_dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "train",
                            transform=transforms.ToTensor()
                          )

# val_dataset = CompoundDataset(
#                             "../../datasets/negev/articles_molecules/preprocess256/aug2",
#                             "../../datasets/negev/not_molecules/preprocess256",
#                             "val",
#                             transform=transforms.ToTensor()
#                           )

cross_val = KFold(n_splits=folds, shuffle=True)

for model_s in models:
    print("")

    for lr in lrs:
        torch.manual_seed(0)
        print("")
        print("---- MODEL:", model_s, " LR:", lr, "----")
        fold_results = np.zeros((4,folds))

        for fold, (train_idx, test_idx) in enumerate(cross_val.split(train_dataset)):
            print("Fold:", fold)

            model = eval(model_s + "()")
            model.to(device)

            train_subset = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subset = torch.utils.data.SubsetRandomSampler(test_idx)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subset)
            test_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_subset)

            train(train_dataloader, model, criterion, nb_epochs, lr, device)
            torch.save(model.state_dict(), "saved_models/"+model_s+"_"+str(lr)+"_"+str(fold)+".pt")

            nb_train_errors = test(train_dataloader, model, device)
            nb_test_errors = test(test_dataloader, model, device)

            fold_results[0,fold] = nb_train_errors
            fold_results[1, fold] = nb_train_errors / len(train_idx) * 100
            fold_results[2,fold] = nb_test_errors
            fold_results[3, fold] = nb_test_errors/len(test_idx)*100
            # print("Total errors predicting test dataset in this fold:", nb_errors)
            # print("Error percentage:", nb_errors/len(test_subset)*100)

        print("")
        print("Cross validation error mean (train split):", np.mean(fold_results[0]))
        print("Cross validation error stdev (train split):", np.std(fold_results[0]))
        print("Cross validation error percentage (%) mean (train split):", np.mean(fold_results[1]), "%")
        print("Cross validation error percentage (%) stdev (train split):", np.std(fold_results[1]), "%")
        print("")
        print("Cross validation error mean (test split):", np.mean(fold_results[2]))
        print("Cross validation error stdev (test split):", np.std(fold_results[2]))
        print("Cross validation error percentage (%) mean (test split):", np.mean(fold_results[3]), "%")
        print("Cross validation error percentage (%) stdev (test split):", np.std(fold_results[3]), "%")

