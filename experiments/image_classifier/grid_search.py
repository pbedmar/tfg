from datasets import CompoundDataset
from functions import train, test
from models import LeNet5, AlexNet, VGG16

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


nb_epochs = 100
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()

models = ["LeNet5", "AlexNet", "VGG16"]
lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
seeds = [1, 2, 3]


train_dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "train",
                            transform=transforms.ToTensor()
                          )

test_dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "test",
                            transform=transforms.ToTensor()
                          )

for model_s in models:

    print("")
    for lr in lrs:

        for seed in seeds:
            print("---- Model:", model_s, " LR:", lr, " Seed:", seed, "----")

            model = eval(model_s + "()")
            model.to(device)

            torch.manual_seed(seed)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

            train(train_dataloader, model, criterion, nb_epochs, lr, device)
            torch.save(model.state_dict(), "saved_models/"+model_s+"_"+str(lr)+"_"+str(seed)+".pt")

            nb_errors = test(test_dataloader, model, device)

            print("Test dataset size:", len(test_dataset))
            print("Total errors predicting test dataset:", nb_errors)
            print("Error percentage:", nb_errors/len(test_dataset)*100)