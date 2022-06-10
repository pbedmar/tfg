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

CONFIG_FILE=configs/custom_vqgan_256_aug2.yaml
CONFIG_NAME=fixseed_256_aug2
SEED=1

for EPOCHS in {70..170..20}
do
    python3 main.py  --name "${CONFIG_NAME}_${EPOCHS}" --max_epochs $EPOCHS --seed $SEED --base $CONFIG_FILE -t True --gpus 0,1,
done


mail -A "slurm-$SLURM_JOBID.out" -s "$SLURM_JOBID $CONFIG_FILE $CONFIG_NAME $EPOCHS ep ha terminado" vicyped@gmail.com <<< "El proceso de ejecución de train_ngpu.sh ha finalizado con los resultados adjuntos"