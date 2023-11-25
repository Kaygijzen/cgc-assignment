#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1

mpirun -np 4 bin/cgc_mpi /var/scratch/bwn200/HPC_data/spring_data_m.npy /var/scratch/bwn200/HPC_data/spring_labels_m_3x20.txt --max-iterations 5