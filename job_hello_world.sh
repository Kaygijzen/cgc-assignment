#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1

mpirun -np 4 hello_world