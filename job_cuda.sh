#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 16
#SBATCH --gres=gpu:1

mpirun -np 16 cgc_cuda /var/scratch/bwn200/HPC_data/spring_data_m.npy /var/scratch/bwn200/HPC_data/spring_labels_m_5x100.txt --max-iterations 125 --output "cuda.txt"