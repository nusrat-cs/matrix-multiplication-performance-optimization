#!/bin/bash
#SBATCH --job-name=cuda-mm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=cuda_output.txt

module purge
module load cuda-toolkit/10.1.243

# use system gcc (compatible)
nvcc cuda_mm.cu -o cuda_mm -O2

./cuda_mm
