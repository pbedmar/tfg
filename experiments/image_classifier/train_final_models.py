from datasets import CompoundDataset
from functions import train, test, dataset_mean_std, init_weights
import models

import sys
import os
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


print = partial(print, flush=True)

# extrae el argumento recibido desde terminal y se fijan las semillas
data_origin = sys.argv[1]
print(data_origin)
torch.manual_seed(1)

# directorio donde se van a almacenar los archivos resultantes de la ejecucion
results_dir = "final_models/"
os.makedirs(results_dir, exist_ok=True)

# si esta disponible, se utiliza la GPU como dispositivo para realizar los entrenamientos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

positive_examples_path = "../../datasets/negev/articles_molecules/preprocess256/aug2"

if data_origin == "original_dataset": # los ejemplos negativos se toman directamente de los del dataset original de la Universidad de Negev
    negative_examples_path = "../../datasets/negev/not_molecules/preprocess256"
    model = models.AlexNet()
    weight_initialization = "Xavier"
    lr = 5e-5
    optimizer = optim.Adam(model.parameters(), lr)
elif data_origin == "synthetic_dataset": # los ejemplos negativos proceden mitad del dataset original mitad de los hard negatives
    negative_examples_path = "../../datasets/negev/not_molecules_plus_synthetic"
    model = models.AlexNet()
    weight_initialization = "Xavier"
    lr = 5e-5
    optimizer = optim.Adam(model.parameters(), lr)
else:
    raise ValueError("Incorrect data origin specified as parameter.")

# carga del split train del dataset y calculo de la media y desviacion tipica
train_dataset = CompoundDataset(
    positive_examples_path,
    negative_examples_path,
    "train",
    transform=transforms.ToTensor()
)
mean, std = dataset_mean_std(train_dataset)

# carga del split train del dataset normalizado
train_dataset = CompoundDataset(
    positive_examples_path,
    negative_examples_path,
    "train",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean[0], mean[1], mean[2]],
                            [std[0], std[1], std[2]])
    ])
)

# carga del split test del dataset normalizado
test_dataset = CompoundDataset(
    positive_examples_path,
    negative_examples_path,
    "test",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean[0], mean[1], mean[2]],
                             [std[0], std[1], std[2]])
    ])
)

# el DataLoader facilita la carga de los datos dividiendolos en batches en orden aleatorio
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# inicialización del modelo y declaracion de la funcion de error a utilizar
model = model.apply(lambda layer: init_weights(layer, weight_initialization))
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
nb_epochs = 100

# entrenamiento del modelo y almacenamiento de este como archivo, junto con el error de entrenamiento
loss = train(train_dataloader, model, criterion, nb_epochs, device, optimizer)
torch.save(model.state_dict(), results_dir+data_origin+"_model.pt")
torch.save(loss, results_dir+data_origin+"_loss.pt")

# computo de errores sobre el split de test
nb_train_errors = test(train_dataloader, model, device)
nb_test_errors = test(test_dataloader, model, device)

# impresión por pantalla de resultados
print("Test dataset size:", len(test_dataset))
print("Total errors over train dataset:", nb_train_errors)
print("Error percentage (%) over train dataset:", nb_train_errors/len(train_dataset)*100)
print("Total errors over test dataset:", nb_test_errors)
print("Error percentage (%) over test dataset:", nb_test_errors/len(test_dataset)*100)