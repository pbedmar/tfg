from datasets import CompoundDataset
from functions import train, test, dataset_mean_std
import models

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32)

model = models.AlexNet()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
nb_epochs = 150
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr)

for child in model.children():
    print(child)

train(train_dataloader, model, criterion, nb_epochs, device, optimizer)
torch.save(model.state_dict(), "saved_models/alexnettry.pt")

nb_errors = test(test_dataloader, model, device)

print("Test dataset size:", len(test_dataset))
print("Total errors predicting test dataset:", nb_errors)
print("Error percentage:", nb_errors/len(test_dataset)*100)