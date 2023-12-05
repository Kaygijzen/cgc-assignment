Load MPI/CUDA modules
> module load openmpi/gcc/64/4.0.2 gcc/9.3.0 cuda11.1/toolkit/11.1.1

## workflow:
On DAS5/6, compile with:

```
mpicc hello_world.c
```

Create file job.sh:

```
#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
mpirun ./a.out
```

Run using:

```
sbatch job.sh
```

### CUDA

Use `sbatch`, `srun`, or `prun` to schedule jobs on the compute nodes

You can also get an interactive login on a compute node:

> srun -N 1 --pty bash

To easily login on a GPU node, first type:
> alias gpurun="srun -N 1 -C TitanX --gres=gpu:1"

Then type:
> gpurun --pty bash