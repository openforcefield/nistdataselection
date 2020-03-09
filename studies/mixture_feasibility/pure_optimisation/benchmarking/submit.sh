#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J bench
#BSUB -W 168:00
#
# Set the output and error output paths.
#BSUB -o  %J.o
#BSUB -e  %J.e
#
# Set any queue options.
#BSUB -q cpuqueue
#BSUB -M 8

. ~/.bashrc

# Use the right conda environment
conda activate forcebalance
python run.py &> server_output.log
