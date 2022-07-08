import glob
import math

import torch
from torch import optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

# dado un directorio de imagenes, genera dos ficheros train y test donde almacena el nombre de
# aquellas imágenes que van a formar parte del conjunto de train y test respectivamente
def gen_metadata(directory, seed=1):
    directory = directory+"/"
    filenames = glob.glob(directory+"*.jpg")
    train, test = train_test_split(filenames, test_size=0.15, random_state=seed)

    with open(directory+"imclass_train.txt", "w") as f:
        for filename in train[:-1]:
            f.write(filename+"\n")
        f.write(train[-1])

    with open(directory+"imclass_test.txt", "w") as f:
        for filename in test[:-1]:
            f.write(filename+"\n")
        f.write(test[-1])

# funcion de entrenamiento de modelos. mediante los argumentos recibe la configuracion
# necesaria, como son los datos de entrenamiento, el propio modelo a entrenar, la funcion
# de error, el numero de epocas, el dispositivo donde se va a entrenar
def train(dataloader, model, criterion, nb_epochs, device, optimizer):
    acc_losses_by_epoch = torch.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        acc_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            acc_loss = acc_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

        acc_losses_by_epoch[epoch] = acc_loss
        if epoch % 10 == 0:
            print("Epoch nb. ", epoch, " -> loss=",acc_loss,sep="")

    return acc_losses_by_epoch

# funcion que dados los datos de test y un modelo ya entrenado, comprueba el numero de
# errores que produce el modelo sobre tal conjunto de datos.
def test(dataset, model, device):
    nb_errors = 0

    for batch_x, batch_y in dataset:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)

        _, predicted_classes = output.max(1)

        for predicted_class, y in zip(predicted_classes, batch_y):
            if predicted_class != y:
                nb_errors = nb_errors + 1

    return nb_errors

# calcula la media y desviacion tipica de un dataset, para luego utilizarlas con
# fines de normalizacion
def dataset_mean_std(dataset):
    train_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(train_dataloader))[0]
    mean = torch.mean(data, dim=(0,2,3))
    std = torch.std(data, dim=(0,2,3))

    return mean, std

# inicializacion de pesos dado un tipo de inicializacion como parametro
def init_weights(layer, initializer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
        if initializer == "Xavier":
            torch.nn.init.xavier_uniform_(layer.weight)
        elif initializer == "He":
            torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        else:
            raise ValueError('The provided weight initializer has not been implemented.')

