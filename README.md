# TFG del Grado en Ingeniería Informática
## Etiquetado de imágenes en química

Con el desarrollo de la informática y del aprendizaje profundo, ramas científicas como la química han aplicado sus métodos dando lugar a lo que se conoce como cheminformatics. Dentro de esta ciencia para los investigadores es importante poseer un modelo que, aplicado sobre imágenes encontradas en publicaciones científicas, les permita clasificar aquellas que presentan estructuras químicas de las que no. Este proyecto propone entrenar un clasificador que separe este tipo de imágenes. Para ello, se llevarán a cabo pruebas con diferentes arquitecturas e hiperparámetros seleccionando los más adecuados. El conjunto de datos de entrenamiento habrá sido previamente refinado y adaptado a esta tarea, habiéndole añadido hard negatives creados a partir de un modelo generativo de forma que se mejore la eficacia del clasificador.

*Palabras clave:* cheminformatics, aprendizaje profundo, balanceo de conjuntos de datos, clasificación binaria, modelos generativos profundos

### Estructura del proyecto
```
repositorio
│   README.md
│
└───datasets/       -> contiene el dataset original así como las transformaciones, data augmentation y hard negatives generados
│
└───experiments/    -> almacena el código del proyecto
│
└───papers/         -> algunas publicaciones utilizadas en el desarrollo del proyecto
│
└───report/         -> memoria en LaTeX
│
└───slides/         -> pases de diapositivas
```

Determinados datasets y modelos ya entrenados no se encuentran incluidos en este repositorio debido a su gran tamaño.

### Dependencias
Algunas de las dependencias más importantes para que el proyecto funcione correctamente son:
```
pytorch-lightning==1.0.8
torch==1.7.0
torchvision==0.8.1
torchtext==0.8.0
omegaconf==2.0.0
einops>=0.3.0
transformers
pillow
test-tube
sklearn
albumentations
jupyter
imgaug
imagemagick
matplotlib
seaborn
numpy
```