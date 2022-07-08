## Experimentación

Para el desarrollo de este TFG se han utilizado bibliotecas ya existentes así como código creado por mí. En este
directorio se encuentran cuatro subdirectorios, cada uno contiene la implementación de experimentos sobre distintas
facetas del proyecto:

- **decimer/** *(discontinued)* contiene pruebas llevadas a cabo sobre la utilidad DECIMER (DECIMER: towards deep
  learning for chemical image recognition, Rajan et al., 2020). Esta permite transformar una imagen de un compuesto
  químico a su respectivo código SMILES. Tras pruebas con diferentes imágenes se demostró que sólo funcionaba con
  imágenes muy concretas, por lo que se descartó para el TFG. Se puede ver la experimentación en el archivo
  decimer/test_decimer.ipynb.
- **image_classifier/** almacena el código del clasificador de imágenes.
- **taming_transformers/** guarda el código del modelo generador de hard negatives.