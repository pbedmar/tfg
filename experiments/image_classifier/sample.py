from datasets import CompoundDataset
from functions import train, test
import models

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "test",
                            transform=transforms.ToTensor()
                          )

test_dataloader = DataLoader(test_dataset, batch_size=64)

model = models.VGG16()
model.load_state_dict(torch.load("saved_models/small_lr_0.005.pt"))
model.to(device)

zeros=0
ones=0
for data, label in test_dataset:
    if label == 0:
        zeros = zeros + 1
    if label == 1:
        ones = ones + 1

print(zeros, ones)
print()

nb_errors = test(test_dataloader, model, device)

print("Test dataset size:", len(test_dataset))
print("Total errors predicting test dataset:", nb_errors)
print("Error percentage:", nb_errors/len(test_dataset)*100)