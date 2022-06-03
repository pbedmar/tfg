from datasets import CompoundDataset
from functions import train, test
import models

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "train",
                            transform=transforms.ToTensor()
                          )

test_dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "val",
                            transform=transforms.ToTensor()
                          )

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=2)

model = models.LeNet5()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
nb_epochs = 150
lr = 1e-4

train(train_dataloader, model, criterion, nb_epochs, lr, device)
torch.save(model.state_dict(), "saved_models/alexnettry.pt")

nb_errors = test(test_dataloader, model, device)

print("Test dataset size:", len(test_dataset))
print("Total errors predicting test dataset:", nb_errors)
print("Error percentage:", nb_errors/len(test_dataset)*100)