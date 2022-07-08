from datasets import CompoundDataset
from functions import train, test, dataset_mean_std
import models

import os
import random
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

print = partial(print, flush=True)

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# carga del split train del dataset para proceder al calculo de la media y desviacion tipica
train_dataset = CompoundDataset(
    "../../datasets/negev/articles_molecules/preprocess256/aug2",
    "../../datasets/negev/not_molecules/preprocess256",
    "train",
    transform=transforms.ToTensor()
)
mean, std = dataset_mean_std(train_dataset)

# carga del split train del dataset normalizado
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

# configuracion general
criterion = torch.nn.CrossEntropyLoss()
nb_epochs = 100
lr = 5e-5

# diferentes tamaños del dataset con los que se van a realizar entrenamientos
number_of_images = [1, 5, 25, 50, 100, 250, 400, 600, 700]
losses = torch.zeros(len(number_of_images), nb_epochs)

for i, n in enumerate(number_of_images):
    torch.manual_seed(1)

    # elige n muestras del dataset de forma aleatoria
    selected_indexes = random.sample(list(range(n)), n)
    train_subdataset = torch.utils.data.Subset(train_dataset, selected_indexes)
    train_dataloader = DataLoader(train_subdataset, batch_size=32, shuffle=False)

    # inicializacion del modelo y del optimizador
    model = models.LeNet5()
    optimizer = optim.Adam(model.parameters(), lr)
    model.to(device)

    # entrenamiento del modelo
    losses[i] = train(train_dataloader, model, criterion, nb_epochs, device, optimizer)

# almacenar el error de entrenamiento para estudiar como decrece
os.makedirs("lenet_convergence_test/", exist_ok=True)
torch.save(losses, "lenet_convergence_test/lenet_losses.pt")