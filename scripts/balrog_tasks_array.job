#!/bin/bash -l
#SBATCH -J balrog_loc
#SBATCH --array=0-8
#
# Number of nodes and MPI tasks per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000MB
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fkunzwei@mpe.mpg.de
#
#SBATCH --time=04:00:00   # run time in h:m:s, up to 24h possible

module purge
export GBMDATA=/ptmp/fkunzwei/gbm_data
export LD_LIBRARY_PATH=$HOME/sw/MultiNest/lib
#module load gcc/9 impi mkl anaconda/3 mpi4py
#module load intel/19.1.2 impi/2019.8 mkl anaconda/3/2020.02 mpi4py gsl
module load intel/19.1.3 impi/2019.9 anaconda/3/2020.02 mpi4py/3.0.3

# avoid overbooking of the cores which might occur via NumPy/MKL threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# date is passed as argument
TRIGGER_INFO_FILE=$1

srun python ${HOME}/scripts/bkg_pipe/run_balrog.py --multi_trigger_info ${TRIGGER_INFO_FILE} --subtasks 9 --index $SLURM_ARRAY_TASK_ID

wait

exit 0
