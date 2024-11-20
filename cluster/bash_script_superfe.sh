#!/bin/bash
# Asignamos un nombre al trabajo
#SBATCH --job-name=pspin
# Definimos el fichero de output si queremos cambiar el nombre por defecto
#SBATCH --output=job_%j_result.log
#SBATCH --error=job_%j_error.log
# Especificamos el numero de nodos y procesadores por nodo que necesitamos
#SBATCH -p GPU_queue
#SBATCH -w gpu11
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     #To specify the number of processes or tasks
#SBATCH --cpus-per-task=1      #To specify the number of threads per process
#SBATCH --mem=1G                #To specify memory. Without it, only 1 job per node 
# SBATCH --array=0-20%5          #To specify an array of jobs. In this case, 21 jobs with 5 jobs running at the same time
# NOTE: Use SLURM_ARRAY_TASK_ID to get the job index in the array (the equivalent of the i in a for loop)
# SLURM_SUBMIT_DIR = The current directory from where a srun or sbatch command is executed. It can be changed with --chdir=<path>

#RESULTS_DIR = The directory where generated files should be copied to before the job is finished
RESULTS_DIR="$SLURM_SUBMIT_DIR/data/${SLURM_JOB_ID}"

mkdir -p $RESULTS_DIR

echo "JOB ID: $SLURM_JOB_ID"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "Copying project files to /scratch partition in node $SLURMD_NODENAME."
cp ~/.julia/environments/v1.10/*.toml /tmp
cp $SLURM_SUBMIT_DIR/* /tmp/
# cp -r stc /tmp to copy the src folder and its content (recursively with -r)
echo "Copy Completed!"
cd /tmp
ls

# # # Load any module or library required by your script
module load Julia 
# module load CUDA

# # Each srun after the first creates a new step for the job
srun -D /tmp julia --project=/tmp template_cluster.jl

Recover the files generated in /scratch partition before cleanup
cp -r /tmp/*.jld2 $RESULTS_DIR # recover all the .jld2 datafiles
cp -r /tmp/*.log $RESULTS_DIR # recover all the .log files (result.log and error.log)

# Unload all modules once the execution is done
module purge
