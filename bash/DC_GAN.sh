#!/bin/sh
#SBATCH --gres=gpu:2
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --time=2-0
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user zcoster@ksu.edu
module load Python/3.10.4-GCCcore-11.3.0
source ~/virtualenvs/dcgan/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python ~/CIS732_Project/DC_GAN.py
