## Datasets

Los directorios que aparecen en este apartado son:

- **negev/** contiene el dataset entregado por los científicos israelíes. Tiene tres subdirectorios, de los cuales
  algunos de ellos pueden tener mas subdirectorios internos, almacenando las transformaciones que se han realizado (
  normalización del tamaño, data augmentation):
    - **articles_molecules/** ejemplos positivos, imágenes de moléculas organometálicas.
    - **not_molecules/** ejemplos negativos.
    - **not_molecules_plus_synthetic** dataset creado por mi con mitad ejemplos negativos originales, mitad hard
      negatives.
- **sample/** contiene imágenes utilizadas en el fichero sampling_experiments.ipynb como entrada al modelo generativo.

Por su tamaño, se almacenan en Google Drive, y se pueden descargar en: https://drive.google.com/drive/folders/1fShMIKozdNFJY3SDFkZ6ivkim3WEEjAG?usp=sharing