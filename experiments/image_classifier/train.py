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
                            "train"
                          )

mean_std_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
for batch_x, batch_y in mean_std_dataloader:
    train_mean = batch_x.mean()
    train_stdev = batch_x.std()

print(train_mean)
print(train_stdev)

train_dataset.transform = transforms.Normalize(mean=train_mean, std=train_stdev)

test_dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "test",
                            transform=transforms.Normalize(mean=train_mean, std=train_stdev)
                          )

print(torch.max(train_dataset[0][0]))

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64)

model = models.VGG16()
model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
nb_epochs = 10
lr = 0.01

train(train_dataloader, model, criterion, nb_epochs, lr, device)
torch.save(model.state_dict(), "saved_models/try.pt")

nb_errors = test(test_dataloader, model, device)

print("Test dataset size:", len(test_dataset))
print("Total errors predicting test dataset:", nb_errors)
print("Error percentage:", nb_errors/len(test_dataset)*100)