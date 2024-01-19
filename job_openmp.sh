#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1

mpirun -np 4 cgc_openmp /var/scratch/bwn200/HPC_data/spring_data_l.npy /var/scratch/bwn200/HPC_data/spring_labels_l_5x100.txt --max-iterations 10  --output "openmp.txt"