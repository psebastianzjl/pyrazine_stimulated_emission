#!/bin/bash -l
#SBATCH --job-name=pyr_r23_t1
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=1000MB
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --partition=test4
#SBATCH --account=test4
#SBATCH -o output.log
source /etc/profile.d/modules.sh


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpc/intel/Oneapi/mkl/2022.0.2/lib/intel64/
#alias python=python3

#export PATH=$PATH:/usr/bin/python3

module load anaconda/4.5.11

# For TURBOMOLE v7.2:
# --nodes           : scale by the size of the calculation
# --ntasks-per-node : always a singel task per node
# --cpus-per-task   : always full nodes

# load modules
module add Turbomole/7.7
module add intel/oneapi

# TURBOMOLE needs this variable
export TURBOTMPDIR=/home/slurm1631488513938763778/md_calc/${SLURM_JOBID}
workdir=${TURBOTMPDIR}
submitdir=${SLURM_SUBMIT_DIR}

# copy files to work
mkdir -p $workdir
cd $workdir
cp -r $submitdir/* $workdir

# set parallel environment
export PARNODES=10 
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# run jobex calculation
#dscf > dscf.out
#ricc2 > ricc2.out


#run nad.exe
echo "#--- Job started at `date`"
./nad.exe > nad.out

# copy files back
#cp -r * $submitdir
rm C*
rm adc*
rm mp2*
rm output.log
cp -r * $submitdir

# clean up
rm -r $workdir
echo "#--- Job ended at `date`"
exit 0
