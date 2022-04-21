#!/bin/bash

#SBATCH --job-name taming-train   # Nombre del proceso
#SBATCH --partition dios          # Cola para ejecutar
#SBATCH --gres=gpu:2             # Numero de gpus a usar
#SBATCH --nodelist=hera

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export HOME="/mnt/homeGPU1/pbedmar/"

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU1/pbedmar/pbedmar.tfg.taming.train/
conda env list
export TFHUB_CACHE_DIR=.

cd /mnt/homeGPU1/pbedmar/pycharm/experiments/taming_transformers/taming-transformers/

EPOCHS=80
CONFIG=configs/custom_vqgan_512_aug_2.yaml

python3 main.py --max_epochs $EPOCHS --base $CONFIG -t True --gpus 0,1,

mail -A "slurm-$SLURM_JOBID.out" -s "$SLURM_JOBID $CONFIG $EPOCHS ep ha terminado" vicyped@gmail.com <<< "El proceso de ejecución de train_ngpu.sh ha finalizado con los resultados adjuntos"