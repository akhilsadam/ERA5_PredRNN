#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** Serial Job in Normal Queue***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch serial.slurm" on a Lonestar6 login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J weather_prediction           # Job name
#SBATCH -o weather_prediction.o%j       # Name of stdout output file
#SBATCH -e weather_prediction.e%j       # Name of stderr error file
#SBATCH -p rtx          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 16:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
### #SBATCH -A myproject       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=akhil.sadam@utexas.edu

# Any other commands must follow all #SBATCH directives...
module load python3/3.9.2
module list
pwd
date

# Launch serial code...
source ../venv/bin/activate
#python3 train.py -m predrnn_v2 predrnn_v2 predrnn_v2 -il 3 5 20 -pn LS6_toy1_control_predrnnTests LS6_toy1_control_predrnnTests LS6_toy1_control_predrnnTests # Do not use ibrun or any other MPI launcher
# python3 train.py -m reZeroTF predrnn_v2 BERT LSTM

# IMPORTANT !
export NCCL_IB_DISABLE=1;
export NCCL_P2P_DISABLE=1;
export NCCL_DEBUG=INFO;
export NCCL_SOCKET_IFNAME=en,eth,em,bond;
export CXX=g++;

accelerate launch train.py -m DMDNet -pre POD_snapshot
#python3 train.py -m DMDNet -pre POD_snapshot -p 4000
