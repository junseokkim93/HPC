#!/bin/bash -x
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --time=30
#SBATCH --mail-user=junseokkim93@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=10gb
#SBATCH --export=ALL
#SBATCH --partition=dev_multiple
#SBATCH --output="out.out"

module purge

module load devel/python/3.9.2_gnu_10.2
module load mpi/openmpi/4.1

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores"
echo "Each node has ${SLURM_MEM_PER_NODE} of memory allocated to this job."

mpirun -n 128 python3 parallelization_LBM.py --Nx=1000 --Ny=1000 --Ndx=8 --Ndy=16 --t=10000
mpirun -n 64 python3 parallelization_LBM.py --Nx=1000 --Ny=1000 --Ndx=8 --Ndy=8 --t=10000
mpirun -n 32 python3 parallelization_LBM.py --Nx=1000 --Ny=1000 --Ndx=4 --Ndy=8 --t=10000
mpirun -n 16 python3 parallelization_LBM.py --Nx=1000 --Ny=1000 --Ndx=4 --Ndy=4 --t=10000
mpirun -n 8 python3 parallelization_LBM.py --Nx=1000 --Ny=1000 --Ndx=2 --Ndy=4 --t=10000
mpirun -n 4 python3 parallelization_LBM.py --Nx=1000 --Ny=1000 --Ndx=2 --Ndy=2 --t=10000
mpirun -n 2 python3 parallelization_LBM.py --Nx=1000 --Ny=1000 --Ndx=1 --Ndy=2 --t=10000
mpirun -n 1 python3 parallelization_LBM.py --Nx=1000 --Ny=1000 --Ndx=1 --Ndy=1 --t=10000
