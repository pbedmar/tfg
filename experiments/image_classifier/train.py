from datasets import CompoundDataset
from functions import train
import models

import torch
from torch.utils.data import DataLoader

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = CompoundDataset(
                            "../../datasets/negev/articles_molecules/preprocess256/aug2",
                            "../../datasets/negev/not_molecules/preprocess256",
                            "train"
                          )

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
model = models.VGG16()
model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
nb_epochs = 150
lr = 0.05

train(dataloader, model, criterion, nb_epochs, lr, device)