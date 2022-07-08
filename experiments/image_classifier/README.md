## Clasificador de imágenes

Aquí se implementa el código del clasificador. El objetivo es que sea capaz de separar aquellas imágenes que contienen
estructuras de moléculas de aquellas que no. Para ello, se declaran los siguientes ficheros:

- **final_models/** contiene los modelos finales que se entregan a los científicos de la Universidad de Negev, así como
  el error de entrenamiento. Resultados de entrenar train_final_models.py

- **lenet_convergence_test/** almacena los errores de entrenamiento obtenidos de ejecutar train_lenet.py.

- **stored_models_original_dataset/** y **stored_results_original_dataset/** almacena los modelos y errores de
  entrenamiento de ejecutar grid_search.py con el parámetro `original_dataset`.

- **stored_models_synthetic_dataset/** y **stored_results_synthetic_dataset/** almacena los modelos y errores de
  entrenamiento de ejecutar grid_search.py con el parámetro `synthetic_dataset`.

- **datasets.py** declara la clase CompoundDataset, una clase que hereda la clase Dataset de Pytorch. Este objeto
  facilita la carga de las imágenes y su uso por las funciones y sentencias incluidas en PyTorch.

- **models.py** implementa cada una de las arquitecturas con las que voy a trabajar (LeNet5, AlexNet y VGG16). Se
  declaran cada una de sus capas, funciones de activación y el orden en el que la información fluye por estas.

- **grid\_search.py** implementa la *grid search*, mediante un parámetro podemos indicar si queremos realizarla
  entrenando los modelos sobre el *dataset* con *hard negatives* o sin ellos.

- **train\_final\_models.py** entrena los modelos finales con la configuración decidida tras realizar la *grid search*.

- **train\_lenet.py** entrena modelos con LeNet5 utilizando diferentes tamaños de *dataset*, para estudiar como estudia
  el tamaño de este al decrecimiento del error de entrenamiento.

- **functions.py** declara funciones utilizadas por todos estos ficheros.

El contenido de los directorios final_models/ y lenet_convergence_test/ se puede descargar
en: https://drive.google.com/drive/folders/14RFuY6r43YgB4izVC8PtwrDVvIVU1SOw?usp=sharing. Para el resto de directorios
no se adjuntan los modelos, ya que al ser una grid search su tamaño es muy grande.

Los archivos ejecutables son train_lenet.py, train_final_models.py y grid_search.py. Se ejecutan directamente desde la
terminal, mediante la orden `pyhton` o `pyhton3`. Los dos últimos scripts deben de ir acompañados de un
parámetro,`original_dataset` o `synthetic_dataset`, indicando si se quieren realizar los experimentos sobre el *dataset*
original o el que contiene los *hard negatives*. Ejemplo:

```
$> python grid_search.py original_dataset
```