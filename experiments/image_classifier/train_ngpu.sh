#!/bin/bash

#SBATCH --job-name image_classifier   # Nombre del proceso
#SBATCH --partition dios          # Cola para ejecutar
#SBATCH --gres=gpu:1             # Numero de gpus a usar
#SBATCH --nodelist=hera

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export HOME="/mnt/homeGPU1/pbedmar/"

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU1/pbedmar/pbedmar.tfg.taming.train/
conda env list
export TFHUB_CACHE_DIR=.

cd /mnt/homeGPU1/pbedmar/pycharm/experiments/image_classifier/

python3 train.py

mail -A "slurm-$SLURM_JOBID.out" -s "ha terminado" vicyped@gmail.com <<< "El proceso de ejecución de train_ngpu.sh ha finalizado con los resultados adjuntos"