## Generador de *hard negatives*

En este subdirectorio se encuentra la implementacion del generador de *hard negatives*. Estos, como se describe en la
memoria, se utilizarán para entrenar un modelo clasificador de forma que sea robusto a ejemplos negativos con parecido a
moléculas. Encontramos diferentes archivos y directorios, cada uno con una función diferente:

- El subdirectorio **taming-transformers/** contiene la implementación del modelo generativo proporcionada por sus
  autores
  *(Taming transformers for high-resolution image synthesis, Esser et al., 2021)*

- El subdirectorio **synthetic_dataset/** contendrá los 400 *hard_negatives* que se generan tras ejecutar
  sample_and_clean_molecules.ipynb

- El subdirectorio **validation/** contendrá las pruebas que se han realizado en sampling_experiment.ipynb para elegir
  el mejor dataset sobre el que entrenar el modelo generador.

- **sampling_experiment.ipynb** es un cuaderno que, una vez entrenados los modelos, permite cargarlos y generar imágenes
  sintéticas a partir de otra imagen de entrada. Se utiliza para comprobar a partir de que dataset se generan mejores
  resultados y con qué número de épocas. Es importante saber que este modelo generativo necesita de una imagen
  condicionante para generar una imagen sintética: se realizarán pruebas con distintos tipos, entre ellas ruido, para
  comprobar con cuáles se obtienen mejores resultados.

- **sample_and_clean_molecules.ipynb**  permite generar un lote de imágenes sintéticas indicando el modelo que queremos
  utilizar y las propiedades del ruido, en concreto de ruido Perlín, ruido que explicaremos en los experimentos. También
  creará el *dataset* final que contiene *hard negatives*.

- **functions.py** declara funciones utilizadas por todos estos cuadernos.

No existen ficheros ejecutables como tal (excepto los ficheros taming-transformers/train_ngpu.sh). Todos los archivos
son Jupyter Notebooks, formados por celdas de texto y código que se pueden ejecutar de forma interativa.

El modelo se entrena mediante los ficheros taming-transformers/train_ngpu.sh. Estos siguen la estructura de un fichero
SLURM, ya que el trabajo será procesado por este sistema de colas en el clúster GPU, permitiendo un entrenamiento mucho
más rápido. En ellos se indica la semilla, el número de épocas con las que se va a entrenar el modelo, el
número de GPUs a utilizar, etc. Para lanzar uno de estos ficheros en el clúster se ejecuta:

```
$> sbatch taming-transformers/train_ngpu.sh
```