import sys
import os
import math
import numpy as np
from functools import partial

from datasets import CompoundDataset
from functions import train, test, dataset_mean_std, init_weights
from models import LeNet5, AlexNet, VGG16

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import KFold

print = partial(print, flush=True)

# extrae el argumento recibido desde terminal y se fijan las semillas
data_origin = sys.argv[1]
torch.manual_seed(1)
np.random.seed(1)

# configuracion general para todos los modelos
nb_epochs = 100
batch_size = 32
folds = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()

# modelos e hiperparametros a utilizar
models = ["LeNet5", "AlexNet", "VGG16"]
lrs = [5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7]
weight_initializations = ["He", "Xavier"]
optimizers = ["SGD", "Adam", "Adadelta"]

positive_examples_path = "../../datasets/negev/articles_molecules/preprocess256/aug2"

if data_origin == "original_dataset": # los ejemplos negativos se toman directamente de los del dataset original de la Universidad de Negev
    negative_examples_path = "../../datasets/negev/not_molecules/preprocess256"
elif data_origin == "synthetic_dataset": # los ejemplos negativos proceden mitad del dataset original mitad de los hard negatives
    negative_examples_path = "../../datasets/negev/not_molecules_plus_synthetic"
else:
    raise ValueError("Incorrect data origin specified as parameter.")

print(data_origin)

# carga del split train del dataset para proceder al calculo de la media y desviacion tipica
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

# tensores donde se registran los resultados para ser guardados tras la ejecucion
losses = torch.zeros(len(models), len(weight_initializations), len(optimizers), len(lrs), folds, nb_epochs)
results = torch.zeros(len(models), len(weight_initializations), len(optimizers), len(lrs), 4, 4)

# bucle grid search
for m, model_s in enumerate(models):
    print("")
    for w, weight_initialization_s in enumerate(weight_initializations):
        print("")
        for o, optimizer_s in enumerate(optimizers):
            print("")
            for l, lr in enumerate(lrs):
                print("")
                print("---- MODEL:", model_s, " WEIGHT_INIT:", weight_initialization_s, " OPTIMIZER:", optimizer_s, " LR:", lr, "----")

                # inicializa folds de la validacion cruzada
                cross_val = KFold(n_splits=folds, shuffle=True)
                fold_results = torch.zeros(4, folds)

                # itera sobre los folds de la validacion cruzada
                for fold, (train_idx, val_idx) in enumerate(cross_val.split(train_dataset)):
                    print("Fold:", fold)

                    # semilla para cada fold
                    torch.manual_seed(fold)
                    np.random.seed(fold)

                    # inicializacion del modelo
                    model = eval(model_s + "()")
                    model.to(device)

                    # inicializacion de sus pesos
                    model = model.apply(lambda layer: init_weights(layer, weight_initialization_s))

                    # inicializacion del optimizador
                    optimizer = eval("optim."+optimizer_s+"(model.parameters(), lr)")

                    # splits del dataset para la validacion cruzada
                    train_subset = torch.utils.data.SubsetRandomSampler(train_idx)
                    val_subset = torch.utils.data.SubsetRandomSampler(val_idx)

                    # el DataLoader facilita la carga de los datos dividiendolos en batches en orden aleatorio
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subset)
                    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subset)

                    # entrenar el modelo con la configuracion y datos correspondientes
                    losses[m,w,o,l,fold] = train(train_dataloader, model, criterion, nb_epochs, device, optimizer)

                    # almacenamiento del modelo creado
                    models_dir = "stored_models_"+data_origin+"/"
                    os.makedirs(models_dir, exist_ok=True)
                    torch.save(model.state_dict(), models_dir+model_s+"_"+weight_initialization_s+"_"+optimizer_s+"_"+str(lr)+"_"+str(fold)+".pt")

                    # computo de errores sobre el split de test
                    nb_train_errors = test(train_dataloader, model, device)
                    nb_test_errors = test(val_dataloader, model, device)

                    # registro de los valores de error, tanto en valor absoluto como relativo
                    fold_results[0, fold] = nb_train_errors
                    fold_results[1, fold] = nb_train_errors / len(train_idx) * 100
                    fold_results[2, fold] = nb_test_errors
                    fold_results[3, fold] = nb_test_errors/len(val_idx)*100

                # calculo de la media, desviacion tipica, maximo y minimo de error entre folds
                for i in range(4):
                    results[m, w, o, l, i, 0] = torch.mean(fold_results[i])
                    results[m, w, o, l, i, 1] = torch.std(fold_results[i])
                    results[m, w, o, l, i, 2] = torch.max(fold_results[i])
                    results[m, w, o, l, i, 3] = torch.min(fold_results[i])


                # impresión por pantalla de resultados
                print("Cross validation error mean (train split):", results[m, w, o, l, 0, 0].item())
                print("Cross validation error stdev (train split):", results[m, w, o, l, 0, 1].item())
                print("Cross validation error max (train split):", results[m, w, o, l, 0, 2].item())
                print("Cross validation error min (train split):", results[m, w, o, l, 0, 3].item())
                print("Cross validation error percentage (%) mean (train split):", results[m, w, o, l, 1, 0].item(), "%")
                print("Cross validation error percentage (%) stdev (train split):", results[m, w, o, l, 1, 1].item(), "%")
                print("Cross validation error percentage (%) max (train split):", results[m, w, o, l, 1, 2].item(), "%")
                print("Cross validation error percentage (%) min (train split):", results[m, w, o, l, 1, 3].item(), "%")
                print("Cross validation error mean (val split):", results[m, w, o, l, 2, 0].item())
                print("Cross validation error stdev (val split):", results[m, w, o, l, 2, 1].item())
                print("Cross validation error max (val split):", results[m, w, o, l, 2, 2].item())
                print("Cross validation error min (val split):", results[m, w, o, l, 2, 3].item())
                print("Cross validation error percentage (%) mean (val split):", results[m, w, o, l, 3, 0].item(), "%")
                print("Cross validation error percentage (%) stdev (val split):", results[m, w, o, l, 3, 1].item(), "%")
                print("Cross validation error percentage (%) max (val split):", results[m, w, o, l, 3, 2].item(), "%")
                print("Cross validation error percentage (%) min (val split):", results[m, w, o, l, 3, 3].item(), "%")

# almacenar error de entrenamiento y errores de clasificacion
results_dir = "stored_results_"+data_origin+"/"
os.makedirs(results_dir, exist_ok=True)
torch.save(losses, results_dir+"grid_search_losses.pt")
torch.save(results, results_dir+"grid_search_results.pt")