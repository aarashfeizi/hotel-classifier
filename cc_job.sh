#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --job-name=siamese-network-lr00006
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=46G
#SBATCH --time=2-0
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aarash.feizi@mail.mcgill.ca

source /home/aarash/venv-siamese/bin/activate


python3 train.py -cuda \
        -dsn cub \
        -dsp ../../dataset/ \
        -sdp images \
        -sp models \
        -gpu 0 \
        -wr 10 \
        -bs 8 \
        -lf 1000 \
        -tf 10000 \
        -sf 10000 \
        -ep 100 \
        -lr 0.00006 \
        -sg \

